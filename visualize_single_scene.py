import argparse
import colorsys
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import habitat_sim
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from config import load_eval_config
from eval import get_closest_dist
from eval.actor import MONActor
from eval.habitat_evaluator import HabitatEvaluator, Result
from eval.semantic_collision import _SEMANTIC_SAFETY_RADIUS_CELLS
from vision_models.coco_classes import COCO_CLASSES


PANEL_SIZE = 640
DEFAULT_CONFIG = "config/mon/eval_conf.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent
NON_VISUAL_SEMANTIC_LABELS = ["floor", "wall", "ceiling"]
DISTINCT_BRG_COLORS = [
    (255, 99, 71),
    (65, 105, 225),
    (60, 179, 113),
    (238, 130, 238),
    (255, 215, 0),
    (0, 206, 209),
    (255, 140, 0),
    (154, 205, 50),
    (220, 20, 60),
    (123, 104, 238),
    (255, 105, 180),
    (0, 191, 255),
    (50, 205, 50),
    (255, 69, 0),
    (186, 85, 211),
    (255, 228, 181),
    (70, 130, 180),
    (46, 139, 87),
    (255, 160, 122),
    (127, 255, 212),
]

# Fixed label→BGR color mapping for obstacle labels (shared by semantic_map and argmax_map)
_OBSTACLE_LABELS_CANONICAL = list(_SEMANTIC_SAFETY_RADIUS_CELLS.keys())
_OBSTACLE_LABEL_COLOR = {
    label: DISTINCT_BRG_COLORS[i % len(DISTINCT_BRG_COLORS)]
    for i, label in enumerate(_OBSTACLE_LABELS_CANONICAL)
}


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run one Habitat episode and save a visualization video."
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="Exact or partial scene_id to select. If omitted, uses --episode-id or the first episode.",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        default=None,
        help="Exact episode_id to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for max evaluation steps.",
    )
    parser.add_argument(
        "--semantic-labels",
        type=str,
        default=None,
        help="Comma-separated labels for the CLIP semantic map. Defaults to scene categories.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.0,
        help="Minimum CLIP similarity required to color a map cell.",
    )
    parser.add_argument(
        "--semantic-coco80",
        action="store_true",
        help="Use all 80 MS-COCO classes for semantic-map similarity.",
    )
    parser.add_argument(
        "--no-semantic-coco80",
        dest="semantic_coco80",
        action="store_false",
        help="Disable COCO80 semantic-map labels and use scene/default labels instead.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/single_scene_debug.mp4",
        help="Output video path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Saved video fps.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live OpenCV playback and only save the video.",
    )
    parser.set_defaults(semantic_coco80=True)
    return parser.parse_known_args()


def unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def select_episode(evaluator: HabitatEvaluator, scene_id: Optional[str], episode_id: Optional[int]):
    if episode_id is not None:
        matches = [ep for ep in evaluator.episodes if ep.episode_id == episode_id]
        if not matches:
            raise ValueError(f"No episode with episode_id={episode_id}.")
        return matches[0]

    if scene_id is not None:
        exact = [ep for ep in evaluator.episodes if ep.scene_id == scene_id]
        if len(exact) == 1:
            return exact[0]
        partial = [ep for ep in evaluator.episodes if scene_id in ep.scene_id]
        if len(partial) == 1:
            return partial[0]
        if len(exact) > 1:
            return exact[0]
        if len(partial) > 1:
            candidates = sorted({ep.scene_id for ep in partial})
            raise ValueError(
                "Scene pattern matched multiple scenes. Please be more specific:\n"
                + "\n".join(candidates[:20])
            )
        raise ValueError(f"No scene matched '{scene_id}'.")

    return evaluator.episodes[0]


def state_to_pose_and_metric_xy(state) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    pose = np.zeros((4,), dtype=np.float32)
    pose[0] = -state.position[2]
    pose[1] = -state.position[0]
    pose[2] = state.position[1]

    orientation = state.rotation
    quat = [orientation.x, orientation.y, orientation.z, orientation.w]
    yaw, _, _ = R.from_quat(quat).as_euler("yxz")
    pose[3] = yaw
    return pose, yaw, (pose[0], pose[1])


def orient_xy_map(map_xy: np.ndarray) -> np.ndarray:
    if map_xy.ndim == 2:
        return np.flip(map_xy.transpose((1, 0)), axis=0)
    if map_xy.ndim == 3:
        return np.flip(map_xy.transpose((1, 0, 2)), axis=0)
    raise ValueError(f"Unsupported map rank for visualization: {map_xy.ndim}")


def px_to_display_xy(point_xy: Tuple[int, int], map_shape: Tuple[int, int], panel_size: int) -> Tuple[int, int]:
    map_h = map_shape[1]
    map_w = map_shape[0]
    x, y = int(point_xy[0]), int(point_xy[1])
    disp_x = x * panel_size / map_w
    disp_y = (map_h - 1 - y) * panel_size / map_h
    return int(round(disp_x)), int(round(disp_y))


def path_to_display(path: List[np.ndarray], map_shape: Tuple[int, int], panel_size: int) -> np.ndarray:
    pts = [px_to_display_xy((int(p[0]), int(p[1])), map_shape, panel_size) for p in path]
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pts, dtype=np.int32)


def build_palette(labels: List[str]) -> np.ndarray:
    """Build a BGR color palette for the given labels.

    Obstacle labels (in _OBSTACLE_LABEL_COLOR) always get their canonical fixed
    color so that semantic_map and argmax_map are visually consistent.
    """
    colors = []
    fallback_idx = 0
    for label in labels:
        if label in _OBSTACLE_LABEL_COLOR:
            colors.append(_OBSTACLE_LABEL_COLOR[label])
        else:
            # Cycle through DISTINCT colors for non-obstacle labels
            while fallback_idx < len(DISTINCT_BRG_COLORS) and DISTINCT_BRG_COLORS[fallback_idx] in _OBSTACLE_LABEL_COLOR.values():
                fallback_idx += 1
            if fallback_idx < len(DISTINCT_BRG_COLORS):
                colors.append(DISTINCT_BRG_COLORS[fallback_idx])
                fallback_idx += 1
            else:
                hue = (len(colors) * 0.618033988749895) % 1.0
                sat = [0.95, 0.8, 0.65][len(colors) % 3]
                val = [1.0, 0.9, 0.8][(len(colors) // 3) % 3]
                r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
                colors.append((int(b * 255), int(g * 255), int(r * 255)))

    return np.asarray(colors, dtype=np.uint8)


@torch.no_grad()
def build_semantic_panel(
    mapper,
    labels: List[str],
    text_features: torch.Tensor,
    semantic_threshold: float,
    display_label_count: int,
) -> np.ndarray:
    feature_map = mapper.one_map.feature_map
    conf_map = mapper.one_map.confidence_map > 0
    n_cells = mapper.one_map.n_cells

    semantic_rgb = np.zeros((n_cells, n_cells, 3), dtype=np.uint8)
    if int(conf_map.sum().item()) == 0:
        return cv2.resize(orient_xy_map(semantic_rgb), (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

    coords = torch.nonzero(conf_map, as_tuple=False)
    feats = feature_map[conf_map]
    sims = feats @ text_features.T
    max_sims, label_ids = torch.max(sims, dim=1)
    valid = (max_sims > semantic_threshold) & (label_ids < display_label_count)
    if int(valid.sum().item()) == 0:
        return cv2.resize(orient_xy_map(semantic_rgb), (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

    coords_np = coords.cpu().numpy()
    palette = build_palette(labels)
    valid_np = valid.cpu().numpy()
    semantic_rgb[coords_np[valid_np, 0], coords_np[valid_np, 1]] = palette[label_ids[valid].cpu().numpy()]

    oriented = orient_xy_map(semantic_rgb)
    return cv2.resize(oriented, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)


@torch.no_grad()
def build_gclip_sim_panel(
    mapper,
    obstacle_labels: List[str],
    panel_size: int = PANEL_SIZE,
) -> Tuple[np.ndarray, dict]:
    """Build a heatmap of max GCLIP cosine similarity to obstacle labels.

    Returns (panel_bgr, stats_dict).
    stats_dict keys: fg_mean, fg_std, bg_mean, bg_std, n_explored, threshold_suggestion
    """
    n_cells = mapper.one_map.n_cells
    stats = {}
    blank = np.zeros((n_cells, n_cells, 3), dtype=np.uint8)

    gclip_feat = getattr(mapper.one_map, "feature_map_gclip", None)
    gclip_conf = getattr(mapper.one_map, "confidence_map_gclip", None)
    if gclip_feat is None or gclip_conf is None:
        return cv2.resize(orient_xy_map(blank), (panel_size, panel_size), interpolation=cv2.INTER_NEAREST), stats

    explored = gclip_conf > 0
    if int(explored.sum().item()) == 0:
        return cv2.resize(orient_xy_map(blank), (panel_size, panel_size), interpolation=cv2.INTER_NEAREST), stats

    # Get GCLIP text features for obstacle labels
    gclip_model = getattr(mapper, "gclip_model", None)
    if gclip_model is None:
        return cv2.resize(orient_xy_map(blank), (panel_size, panel_size), interpolation=cv2.INTER_NEAREST), stats

    text_feats = gclip_model.get_text_features(
        [f"a {lbl}" for lbl in obstacle_labels]
    ).to(gclip_feat.device)

    feats = gclip_feat[explored]  # [M, 512]
    import torch.nn.functional as F
    feats_norm = F.normalize(feats, dim=1)
    sims = feats_norm @ text_feats.T  # [M, N_labels]
    max_sims, _ = sims.max(dim=1)  # [M]
    max_sims_np = max_sims.cpu().numpy()

    # Build heatmap: map sim values to color
    sim_map = np.zeros((n_cells, n_cells), dtype=np.float32)
    coords = torch.nonzero(explored, as_tuple=False).cpu().numpy()
    sim_map[coords[:, 0], coords[:, 1]] = max_sims_np

    # Use depth obstacle map as fg/bg proxy
    nav_map = mapper.one_map.navigable_map.astype(bool)
    explored_np = explored.cpu().numpy()
    fg_mask = explored_np & (~nav_map)  # obstacle cells (depth-based)
    bg_mask = explored_np & nav_map     # free cells

    fg_sims = sim_map[fg_mask] if fg_mask.any() else np.array([])
    bg_sims = sim_map[bg_mask] if bg_mask.any() else np.array([])

    stats["n_explored"] = int(explored.sum().item())
    if len(fg_sims) > 0:
        stats["fg_mean"] = float(np.mean(fg_sims))
        stats["fg_std"] = float(np.std(fg_sims))
        stats["fg_median"] = float(np.median(fg_sims))
    if len(bg_sims) > 0:
        stats["bg_mean"] = float(np.mean(bg_sims))
        stats["bg_std"] = float(np.std(bg_sims))
        stats["bg_median"] = float(np.median(bg_sims))
    if len(fg_sims) > 0 and len(bg_sims) > 0:
        # Suggest τ = bg_mean + 1*bg_std (conservative)
        stats["tau_bg_1sigma"] = float(np.mean(bg_sims) + np.std(bg_sims))
        stats["tau_bg_2sigma"] = float(np.mean(bg_sims) + 2 * np.std(bg_sims))
        # d-prime
        pooled_std = np.sqrt((np.std(fg_sims)**2 + np.std(bg_sims)**2) / 2)
        if pooled_std > 1e-8:
            stats["d_prime"] = float((np.mean(fg_sims) - np.mean(bg_sims)) / pooled_std)

    # Colorize: auto-scale to actual sim range for spatial contrast
    explored_sims = sim_map[explored_np]
    vmin = float(explored_sims.min()) - 0.002
    vmax = float(explored_sims.max()) + 0.002
    sim_clipped = np.clip(sim_map, vmin, vmax)
    sim_normalized = ((sim_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(sim_normalized, cv2.COLORMAP_JET)
    # Black out unexplored
    heatmap[~explored_np] = 0

    oriented = orient_xy_map(heatmap)
    panel = cv2.resize(oriented, (panel_size, panel_size), interpolation=cv2.INTER_NEAREST)

    # Overlay stats text
    y = 28
    for key in ["fg_mean", "fg_std", "bg_mean", "bg_std", "d_prime", "tau_bg_1sigma"]:
        if key in stats:
            txt = f"{key}: {stats[key]:.4f}"
            cv2.putText(panel, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20

    return panel, stats


def build_ground_truth_semantic_panel(evaluator: HabitatEvaluator, scene_id: str, query_label: str) -> np.ndarray:
    collision_data = evaluator.get_semantic_collision_data(
        scene_id, query_label, floor_y=getattr(evaluator, "_episode_floor_y", None))
    panel = np.zeros((evaluator.mapping.n_points, evaluator.mapping.n_points, 3), dtype=np.uint8)
    if collision_data.labels:
        palette = build_palette(collision_data.labels)
        for idx in range(1, len(collision_data.labels) + 1):
            panel[collision_data.label_map == idx] = palette[idx - 1]
    oriented = orient_xy_map(panel)
    return cv2.resize(oriented, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)


def build_obstacle_layers_panel(mapper) -> np.ndarray:
    """Combined obstacle panel showing all active layers.

    Layer order (bottom → top, each overwrites previous):
      black      : unexplored / navigable free (default)
      light grey : depth point-cloud obstacle
      red        : YOLO dilated obstacle mask
      orange     : YOLO raw seed points
      purple     : YOLO-CP / CLIP-CP dilated obstacle mask
      yellow     : YOLO-CP / CLIP-CP conf seeds
    """
    n = mapper.one_map.n_cells
    panel = np.zeros((n, n, 3), dtype=np.uint8)       # start black (match other panels)

    base_nav = mapper.one_map.navigable_map.astype(bool)
    panel[~base_nav] = (230, 230, 230)                # depth obstacle → light grey

    active_layers = []  # (label, bgr_color) for legend

    yolo_map = getattr(mapper, "yolo_obstacle_map", None)
    if yolo_map is not None and getattr(mapper, "use_yolo_obstacle_map", False):
        yolo_mask = yolo_map.get_obstacle_mask()
        panel[yolo_mask & base_nav] = (0, 0, 200)     # YOLO dilated → red
        for _frame, _label, px, py in yolo_map._events:
            panel[px, py] = (0, 140, 255)             # YOLO seeds → orange
        active_layers += [("YOLO dilated", (0, 0, 200)), ("YOLO seeds", (0, 140, 255))]

    yolo_cp_map = getattr(mapper, "yolo_cp_obstacle_map", None)
    if yolo_cp_map is not None and getattr(mapper, "use_yolo_cp_obstacle_map", False):
        cp_mask = yolo_cp_map.get_obstacle_mask()
        panel[cp_mask & base_nav] = (180, 0, 180)     # YOLO-CP dilated → purple
        cp_seeds = (yolo_cp_map._conf_map.max(axis=2) >= yolo_cp_map.threshold)
        panel[cp_seeds & base_nav] = (0, 200, 255)    # YOLO-CP seeds → yellow
        active_layers += [("YOLO-CP dilated", (180, 0, 180)), ("YOLO-CP seeds", (0, 200, 255))]

    clip_cp_map = getattr(mapper, "clip_cp_obstacle_map", None)
    use_clip_cp = (getattr(mapper, "use_clip_cp_obstacle_map", False) or
                   getattr(mapper, "use_clip_argmax_obstacle_map", False))
    if clip_cp_map is not None and use_clip_cp:
        cp_mask = clip_cp_map.get_obstacle_mask()
        panel[cp_mask & base_nav] = (180, 0, 180)     # CLIP-CP dilated → purple
        cp_seeds = (clip_cp_map._seed_map > 0)
        panel[cp_seeds & base_nav] = (0, 200, 255)    # CLIP-CP seeds → yellow
        active_layers += [("CLIP-CP dilated", (180, 0, 180)), ("CLIP-CP seeds", (0, 200, 255))]

    panel = cv2.resize(orient_xy_map(panel), (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

    # Draw legend (top-right)
    legend_items = [("free / unexplored", (0, 0, 0)),
                    ("depth obstacle", (230, 230, 230))] + active_layers
    row_h = 18
    lg_w = 200
    lg_h = len(legend_items) * row_h + 10
    x0 = PANEL_SIZE - lg_w - 6
    y0 = PANEL_SIZE - lg_h - 6
    cv2.rectangle(panel, (x0, y0), (x0 + lg_w, y0 + lg_h), (40, 40, 40), -1)
    for i, (lbl, color) in enumerate(legend_items):
        y = y0 + 8 + i * row_h
        cv2.rectangle(panel, (x0 + 6, y), (x0 + 22, y + row_h - 6), color, -1)
        cv2.rectangle(panel, (x0 + 6, y), (x0 + 22, y + row_h - 6), (200, 200, 200), 1)
        cv2.putText(panel, lbl, (x0 + 28, y + row_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def build_obstacle_panel(navigable_map: np.ndarray) -> np.ndarray:
    dilated_obstacles = (~navigable_map.astype(bool)).astype(np.uint8)
    panel = np.zeros((dilated_obstacles.shape[0], dilated_obstacles.shape[1], 3), dtype=np.uint8)
    panel[dilated_obstacles > 0] = np.array([230, 230, 230], dtype=np.uint8)
    panel = orient_xy_map(panel)
    return cv2.resize(panel, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)


def draw_path_robot_goal(
    panel: np.ndarray,
    map_shape: Tuple[int, int],
    robot_px: Tuple[int, int],
    path,
    chosen_detection,
    title: str,
) -> np.ndarray:
    canvas = panel.copy()

    if isinstance(path, list) and len(path) > 1:
        path_pts = path_to_display(path, map_shape, PANEL_SIZE)
        cv2.polylines(canvas, [path_pts], isClosed=False, color=(255, 64, 64), thickness=2)

    robot_disp = px_to_display_xy(robot_px, map_shape, PANEL_SIZE)
    cv2.circle(canvas, robot_disp, 6, (0, 0, 255), -1)

    if chosen_detection is not None:
        goal_disp = px_to_display_xy(chosen_detection, map_shape, PANEL_SIZE)
        cv2.circle(canvas, goal_disp, 7, (0, 255, 0), 2)

    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_legend(
    panel: np.ndarray,
    labels: List[str],
    legend_labels: Optional[List[str]] = None,
    max_items: int = 15,
) -> np.ndarray:
    canvas = panel.copy()
    palette = build_palette(labels)
    color_map = {label: tuple(int(v) for v in palette[idx].tolist()) for idx, label in enumerate(labels)}
    labels_to_draw = legend_labels if legend_labels is not None else labels
    y = 42
    for idx, label in enumerate(labels_to_draw):
        if idx >= max_items:
            cv2.putText(
                canvas,
                f"... +{len(labels_to_draw) - max_items} more",
                (14, y + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            break
        if y > canvas.shape[0] - 16:
            break
        color = color_map[label]
        cv2.rectangle(canvas, (14, y - 12), (34, y + 8), color, -1)
        cv2.putText(canvas, label, (42, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return canvas


def build_rgb_panel(rgb_obs: np.ndarray, info_lines: List[str]) -> np.ndarray:
    rgb = rgb_obs[:, :, :3]
    panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel = cv2.resize(panel, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_AREA)
    y = 28
    for line in info_lines:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return panel


def classify_result(
    evaluator: HabitatEvaluator,
    scene_id: str,
    current_obj: str,
    called_found: bool,
    poses: List[np.ndarray],
) -> Result:
    collided, _ = evaluator.check_semantic_collision(scene_id, current_obj)
    if collided:
        return Result.SEMANTIC_COLLISION

    if called_found:
        dist = get_closest_dist(
            evaluator.sim.get_agent(0).get_state().position[[0, 2]],
            evaluator.scene_data[scene_id].object_locations[current_obj],
            evaluator.is_gibson,
        )
        if dist < evaluator.max_dist:
            return Result.SUCCESS
        pos = evaluator.actor.mapper.chosen_detection
        pos_metric = evaluator.actor.mapper.one_map.px_to_metric(pos[0], pos[1])
        dist_detect = get_closest_dist(
            [-pos_metric[1], -pos_metric[0]],
            evaluator.scene_data[scene_id].object_locations[current_obj],
            evaluator.is_gibson,
        )
        if dist_detect < evaluator.max_dist:
            return Result.FAILURE_NOT_REACHED
        return Result.FAILURE_MISDETECT

    result = Result.FAILURE_OOT
    if len(poses) >= 10 and np.linalg.norm(poses[-1] - poses[-10]) < 0.05:
        result = Result.FAILURE_STUCK
    if result in (Result.FAILURE_OOT, Result.FAILURE_STUCK) and len(evaluator.actor.mapper.nav_goals) == 0:
        result = Result.FAILURE_ALL_EXPLORED
    return result


def resolve_semantic_labels(
    evaluator: HabitatEvaluator,
    episode,
    override: Optional[str],
    use_coco80: bool,
) -> List[str]:
    if use_coco80:
        return list(COCO_CLASSES)

    if override:
        labels = [label.strip() for label in override.split(",") if label.strip()]
        return unique_keep_order(labels)

    scene_labels = sorted(evaluator.scene_data[episode.scene_id].object_locations.keys())
    return unique_keep_order(episode.obj_sequence + scene_labels)


def build_semantic_candidate_labels(display_labels: List[str]) -> List[str]:
    return unique_keep_order(display_labels + NON_VISUAL_SEMANTIC_LABELS)


def build_legend_labels(
    evaluator: HabitatEvaluator,
    labels: List[str],
    max_items: int = 15,
) -> List[str]:
    alias_to_coco = {
        "plant": "potted plant",
        "sofa": "couch",
        "tv_monitor": "tv",
    }

    counts = Counter()
    for ep in evaluator.episodes:
        for obj in ep.obj_sequence:
            counts[alias_to_coco.get(obj, obj)] += 1

    ranked = sorted(labels, key=lambda label: (-counts.get(label, 0), label))
    return ranked[:max_items]


def resolve_eval_paths(eval_conf) -> None:
    object_nav_path = Path(eval_conf.object_nav_path)
    if not object_nav_path.is_absolute():
        object_nav_path = (PROJECT_ROOT / object_nav_path).resolve()
    object.__setattr__(
        eval_conf,
        "object_nav_path",
        str(object_nav_path) + ("/" if not str(object_nav_path).endswith("/") else ""),
    )

    scene_path = Path(eval_conf.scene_path)
    if not scene_path.is_absolute():
        scene_path = (PROJECT_ROOT / scene_path).resolve()

    dataset_cfg = scene_path / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"
    if not dataset_cfg.exists():
        fallback_candidates = [
            (PROJECT_ROOT / "datasets" / "versioned_data" / "hm3d-1.0").resolve(),
            (PROJECT_ROOT / "datasets" / "versioned_data").resolve(),
        ]
        for fallback in fallback_candidates:
            fallback_cfg = fallback / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"
            if fallback_cfg.exists():
                scene_path = fallback
                break

    object.__setattr__(
        eval_conf,
        "scene_path",
        str(scene_path) + ("/" if not str(scene_path).endswith("/") else ""),
    )


def build_output_path(output_arg: str, episode_id: int) -> Path:
    output_path = Path(output_arg)
    run_date = datetime.now().strftime("%Y%m%d")

    if output_path.suffix:
        stem = output_path.stem
        suffix = output_path.suffix
        filename = f"{stem}_ep{episode_id}_{run_date}{suffix}"
        return output_path.with_name(filename)

    filename = f"single_scene_debug_ep{episode_id}_{run_date}.mp4"
    return output_path / filename


def main() -> None:
    args, spock_args = parse_args()
    if not any(arg == "-c" or arg == "--config" for arg in spock_args):
        spock_args = ["-c", DEFAULT_CONFIG] + spock_args
    sys.argv = [sys.argv[0]] + spock_args

    cfg = load_eval_config()
    resolve_eval_paths(cfg.EvalConf)
    evaluator = HabitatEvaluator(cfg.EvalConf, MONActor(cfg.EvalConf))
    if args.max_steps is not None:
        evaluator.max_steps = args.max_steps

    episode = select_episode(evaluator, args.scene_id, args.episode_id)
    evaluator.load_scene(episode.scene_id)
    evaluator.sim.initialize_agent(0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
    evaluator.actor.reset()

    current_obj = episode.obj_sequence[0]
    evaluator.actor.set_query(current_obj)
    evaluator._episode_floor_y = float(episode.start_position[1])
    ground_truth_collision_data = evaluator.get_semantic_collision_data(
        episode.scene_id, current_obj, floor_y=evaluator._episode_floor_y)
    # Pass GT label map to navigator for ACI calibration
    evaluator.actor.mapper.set_gt_label_map(
        ground_truth_collision_data.label_map, ground_truth_collision_data.labels
    )
    semantic_labels = resolve_semantic_labels(
        evaluator,
        episode,
        args.semantic_labels,
        args.semantic_coco80,
    )
    semantic_candidate_labels = build_semantic_candidate_labels(semantic_labels)
    legend_labels = build_legend_labels(evaluator, semantic_labels, max_items=15)
    semantic_text_features = evaluator.actor.mapper.model.get_text_features(
        [f"a {label}" for label in semantic_candidate_labels]
    ).to(evaluator.actor.mapper.one_map.map_device)

    out_path = build_output_path(args.output, episode.episode_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    poses = []
    result = Result.FAILURE_OOT
    tau_history: List[float] = []   # τ per step for YOLO-CP

    try:
        for step in range(evaluator.max_steps):
            observations = evaluator.sim.get_sensor_observations()
            observations["state"] = evaluator.sim.get_agent(0).get_state()

            pose, yaw, metric_xy = state_to_pose_and_metric_xy(observations["state"])
            poses.append(pose)
            robot_px = evaluator.actor.mapper.one_map.metric_to_px(*metric_xy)

            action, called_found = evaluator.actor.act(observations)
            path = evaluator.actor.mapper.path
            chosen_detection = evaluator.actor.mapper.chosen_detection

            # Compute active navigable map once per step (avoids repeated get_obstacle_mask() / dilate calls)
            active_nav = (
                evaluator.actor.mapper.get_active_navigable_map()
                if hasattr(evaluator.actor.mapper, "get_active_navigable_map")
                else evaluator.actor.mapper.one_map.navigable_map
            )
            map_shape = active_nav.shape

            rgb_panel = build_rgb_panel(
                observations["rgb"],
                [
                    f"scene: {episode.scene_id.split('/')[-1]}",
                    f"episode: {episode.episode_id}",
                    f"query: {current_obj}",
                    f"step: {step}",
                    f"yaw: {yaw:.2f}",
                ],
            )
            _half = PANEL_SIZE // 2
            _left_bgr = cv2.cvtColor(observations["rgb_left"][:, :, :3], cv2.COLOR_RGB2BGR)
            _right_bgr = cv2.cvtColor(observations["rgb_right"][:, :, :3], cv2.COLOR_RGB2BGR)
            left_half = cv2.resize(_left_bgr, (PANEL_SIZE, _half), interpolation=cv2.INTER_AREA)
            right_half = cv2.resize(_right_bgr, (PANEL_SIZE, _half), interpolation=cv2.INTER_AREA)
            cv2.putText(left_half, "Left", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right_half, "Right", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            side_views = np.concatenate([left_half, right_half], axis=0)
            obstacle_panel = build_obstacle_panel(active_nav)
            obstacle_title = (
                "Semantic-Inflated Navigable Map + A*"
                if getattr(evaluator.actor.mapper, "use_clip_semantic_nav_map", False)
                else "Dilated Obstacles + A*"
            )
            obstacle_panel = draw_path_robot_goal(
                obstacle_panel,
                map_shape,
                robot_px,
                path,
                chosen_detection,
                obstacle_title,
            )
            semantic_panel = build_semantic_panel(
                evaluator.actor.mapper,
                semantic_labels,
                semantic_text_features,
                args.semantic_threshold,
                len(semantic_labels),
            )
            semantic_panel = draw_path_robot_goal(
                semantic_panel,
                map_shape,
                robot_px,
                path,
                chosen_detection,
                "CLIP Semantic Map",
            )
            semantic_panel = draw_legend(
                semantic_panel,
                semantic_labels,
                legend_labels=legend_labels,
                max_items=15,
            )
            gt_semantic_panel = build_ground_truth_semantic_panel(
                evaluator,
                episode.scene_id,
                current_obj,
            )
            gt_semantic_panel = draw_path_robot_goal(
                gt_semantic_panel,
                map_shape,
                robot_px,
                path,
                chosen_detection,
                "GT Semantic Safety Map",
            )
            gt_semantic_panel = draw_legend(
                gt_semantic_panel,
                ground_truth_collision_data.labels,
                max_items=15,
            )

            yolo_cp_map = getattr(evaluator.actor.mapper, "yolo_cp_obstacle_map", None)
            if yolo_cp_map is not None and getattr(evaluator.actor.mapper, "use_yolo_cp_obstacle_map", False):
                from eval.semantic_collision import metric_to_px as _metric_to_px
                _cell_size = evaluator.mapping.size / evaluator.mapping.n_points
                _rx = -observations["state"].position[2]
                _ry = -observations["state"].position[0]
                _rpx, _rpy = _metric_to_px(_rx, _ry, evaluator.mapping.n_points, _cell_size)
                _cd = evaluator.get_semantic_collision_data(episode.scene_id, current_obj, floor_y=getattr(evaluator, "_episode_floor_y", None))
                yolo_cp_map.calibrate_with_gt(_cd.label_map, _cd.labels, _rpx, _rpy, yaw)
                tau_history.append(yolo_cp_map.threshold)

            use_yolo = getattr(evaluator.actor.mapper, "use_yolo_obstacle_map", False)
            use_cp = getattr(evaluator.actor.mapper, "use_yolo_cp_obstacle_map", False)
            use_clip_cp = (getattr(evaluator.actor.mapper, "use_clip_cp_obstacle_map", False) or
                           getattr(evaluator.actor.mapper, "use_clip_argmax_obstacle_map", False))
            has_gclip = getattr(evaluator.actor.mapper, "gclip_model", None) is not None

            # Build GCLIP similarity heatmap panel if dual-model active
            gclip_panel = None
            if has_gclip:
                _obs_labels = ["chair", "potted plant", "toilet"]
                gclip_panel, gclip_stats = build_gclip_sim_panel(
                    evaluator.actor.mapper, _obs_labels
                )
                gclip_panel = draw_path_robot_goal(
                    gclip_panel, map_shape, robot_px, path,
                    chosen_detection, "GCLIP Obstacle Sim Heatmap",
                )
                # Collect stats for end-of-episode summary
                if step == 0:
                    all_gclip_stats = []
                if gclip_stats:
                    all_gclip_stats.append(gclip_stats)

            if use_yolo or use_cp or use_clip_cp:
                layers_panel = build_obstacle_layers_panel(evaluator.actor.mapper)
                layers_panel = draw_path_robot_goal(
                    layers_panel,
                    map_shape,
                    robot_px,
                    path,
                    chosen_detection,
                    "Obstacle Layers (grey=depth, purple=CLIP-CP)",
                )
                if gclip_panel is not None:
                    # 2 rows: top=[side_views, rgb, layers, gclip_sim], bottom=[side_views, obstacle, semantic, gt]
                    top_row = np.concatenate([side_views, rgb_panel, layers_panel, gclip_panel], axis=1)
                    bottom_row = np.concatenate([np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8), obstacle_panel, semantic_panel, gt_semantic_panel], axis=1)
                    frame = np.concatenate([top_row, bottom_row], axis=0)
                else:
                    frame = np.concatenate([side_views, rgb_panel, layers_panel, semantic_panel, gt_semantic_panel], axis=1)
            elif gclip_panel is not None:
                # No CP layers but GCLIP is active — show gclip sim panel
                top_row = np.concatenate([side_views, rgb_panel, obstacle_panel, gclip_panel], axis=1)
                bottom_row = np.concatenate([np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8), np.zeros_like(rgb_panel), semantic_panel, gt_semantic_panel], axis=1)
                frame = np.concatenate([top_row, bottom_row], axis=0)
            else:
                frame = np.concatenate([side_views, rgb_panel, obstacle_panel, semantic_panel, gt_semantic_panel], axis=1)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (frame.shape[1], frame.shape[0]))

            writer.write(frame)
            if not args.no_display:
                cv2.imshow("single-scene-visualizer", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    result = classify_result(evaluator, episode.scene_id, current_obj, False, poses)
                    break

            evaluator.execute_action(action)

            collided, cause_label = evaluator.check_semantic_collision(episode.scene_id, current_obj)
            if collided:
                result = Result.SEMANTIC_COLLISION
                position = evaluator.sim.get_agent(0).get_state().position
                print(f"SEMANTIC_COLLISION at step={step} cause={cause_label} pos=({-position[2]:.2f}, {-position[0]:.2f}) world_y={position[1]:.2f}")
                break

            if called_found:
                result = classify_result(evaluator, episode.scene_id, current_obj, True, poses)
                break

            # Early stuck detection (matches habitat_evaluator.py):
            # every 100 steps, if displacement over last 100 steps < 0.1m → abort
            if step > 0 and step % 100 == 0 and len(poses) >= 100:
                _recent = np.array(poses[-100:])
                _disp = float(np.linalg.norm(_recent[-1, :2] - _recent[0, :2]))
                if _disp < 0.1:
                    result = Result.FAILURE_STUCK
                    print(f"Early stuck: displacement={_disp:.4f}m over last 100 steps, aborting at step {step}")
                    break
        else:
            result = classify_result(evaluator, episode.scene_id, current_obj, False, poses)

        # Post-loop reclassification (matches habitat_evaluator.py):
        # OOT + barely moved → STUCK
        poses_arr = np.array(poses)
        if result == Result.FAILURE_OOT and len(poses_arr) >= 10:
            if np.linalg.norm(poses_arr[-1] - poses_arr[-10]) < 0.05:
                result = Result.FAILURE_STUCK
        # STUCK or OOT + no frontiers → ALL_EXPLORED
        num_frontiers = len(evaluator.actor.mapper.nav_goals)
        if result in (Result.FAILURE_STUCK, Result.FAILURE_OOT) and num_frontiers == 0:
            result = Result.FAILURE_ALL_EXPLORED
    finally:
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if evaluator.sim is not None:
            evaluator.sim.close()

    print(f"Saved video to: {out_path}")
    print(f"Scene: {episode.scene_id}")
    print(f"Episode: {episode.episode_id}")
    print(f"Query: {current_obj}")
    print(f"Result: {result.name}")

    # Print and save GCLIP sim distribution
    if 'all_gclip_stats' in dir() and all_gclip_stats:
        final_stats = all_gclip_stats[-1]
        print("\n=== GCLIP Obstacle Similarity Stats (final step) ===")
        for k, v in final_stats.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        # Save histogram using GT semantic labels for proper fg/bg
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import torch.nn.functional as _F

            mapper = evaluator.actor.mapper
            gclip_feat = mapper.one_map.feature_map_gclip
            gclip_conf = mapper.one_map.confidence_map_gclip
            if gclip_feat is not None and gclip_conf is not None:
                explored = gclip_conf > 0
                if explored.any():
                    _obs_labels = ["chair", "potted plant", "toilet"]
                    text_feats = mapper.gclip_model.get_text_features(
                        [f"a {l}" for l in _obs_labels]
                    ).to(gclip_feat.device)
                    feats = gclip_feat[explored]
                    feats_n = _F.normalize(feats, dim=1)
                    sims = feats_n @ text_feats.T  # [M, N_labels]
                    max_sims = sims.max(dim=1)[0].cpu().numpy()
                    per_label_sims = sims.cpu().numpy()  # [M, N_labels]

                    explored_np = explored.cpu().numpy()
                    coords = torch.nonzero(explored, as_tuple=False).cpu().numpy()

                    # Use GT semantic collision map for true fg/bg
                    gt_collision = evaluator.get_semantic_collision_data(episode.scene_id, current_obj, floor_y=getattr(evaluator, "_episode_floor_y", None))
                    gt_label_map = gt_collision.label_map  # [n, n] with 0=bg, 1+=obstacle class
                    gt_fg = gt_label_map > 0  # true semantic obstacle cells
                    gt_bg = (gt_label_map == 0) & explored_np  # explored non-obstacle cells

                    fg_indices = gt_fg[coords[:, 0], coords[:, 1]]
                    bg_indices = gt_bg[coords[:, 0], coords[:, 1]]

                    print(f"\n=== GCLIP Stats with GT fg/bg ===")
                    if fg_indices.any():
                        fg_sims = max_sims[fg_indices]
                        print(f"  GT_FG: n={fg_indices.sum()}, mean={fg_sims.mean():.4f}, std={fg_sims.std():.4f}, median={np.median(fg_sims):.4f}")
                    if bg_indices.any():
                        bg_sims = max_sims[bg_indices]
                        print(f"  GT_BG: n={bg_indices.sum()}, mean={bg_sims.mean():.4f}, std={bg_sims.std():.4f}, median={np.median(bg_sims):.4f}")
                    if fg_indices.any() and bg_indices.any():
                        pooled = np.sqrt((fg_sims.std()**2 + bg_sims.std()**2) / 2)
                        if pooled > 1e-8:
                            dp = (fg_sims.mean() - bg_sims.mean()) / pooled
                            print(f"  d-prime (GT): {dp:.4f}")
                        tau_1s = bg_sims.mean() + bg_sims.std()
                        tau_2s = bg_sims.mean() + 2 * bg_sims.std()
                        print(f"  tau_bg+1sigma: {tau_1s:.4f}")
                        print(f"  tau_bg+2sigma: {tau_2s:.4f}")

                    # Per-label breakdown
                    for j, lbl in enumerate(_obs_labels):
                        label_sims = per_label_sims[:, j]
                        # GT cells for this specific label
                        if lbl in gt_collision.labels:
                            gt_idx = gt_collision.labels.index(lbl) + 1
                            lbl_fg = (gt_label_map == gt_idx)
                            lbl_fg_indices = lbl_fg[coords[:, 0], coords[:, 1]]
                            if lbl_fg_indices.any():
                                lbl_fg_sims = label_sims[lbl_fg_indices]
                                lbl_bg_sims = label_sims[bg_indices]
                                print(f"  [{lbl}] GT_FG: n={lbl_fg_indices.sum()}, sim mean={lbl_fg_sims.mean():.4f}±{lbl_fg_sims.std():.4f} | BG mean={lbl_bg_sims.mean():.4f}±{lbl_bg_sims.std():.4f}")

                    # Plot: 2 subplots - depth-proxy vs GT
                    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
                    bins = np.linspace(-0.05, 0.45, 80)

                    # Left: depth-based (old, for reference)
                    nav_map = mapper.one_map.navigable_map.astype(bool)
                    depth_fg = explored_np & (~nav_map)
                    depth_bg = explored_np & nav_map
                    dfg_idx = depth_fg[coords[:, 0], coords[:, 1]]
                    dbg_idx = depth_bg[coords[:, 0], coords[:, 1]]
                    ax = axes[0]
                    if dbg_idx.any():
                        ax.hist(max_sims[dbg_idx], bins=bins, alpha=0.6, label=f"BG-depth (n={dbg_idx.sum()})", color="steelblue")
                    if dfg_idx.any():
                        ax.hist(max_sims[dfg_idx], bins=bins, alpha=0.6, label=f"FG-depth (n={dfg_idx.sum()})", color="coral")
                    ax.set_title("Depth-based fg/bg (noisy)")
                    ax.set_xlabel("Max cosine sim")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    # Right: GT-based
                    ax = axes[1]
                    if bg_indices.any():
                        ax.hist(max_sims[bg_indices], bins=bins, alpha=0.6, label=f"BG-GT (n={bg_indices.sum()})", color="steelblue")
                    if fg_indices.any():
                        ax.hist(max_sims[fg_indices], bins=bins, alpha=0.6, label=f"FG-GT (n={fg_indices.sum()})", color="coral")
                    if fg_indices.any() and bg_indices.any():
                        ax.axvline(tau_1s, color="green", linestyle="--", label=f"bg+1σ={tau_1s:.3f}")
                        ax.axvline(tau_2s, color="orange", linestyle="--", label=f"bg+2σ={tau_2s:.3f}")
                    ax.set_title(f"GT semantic fg/bg | d'={dp:.2f}" if fg_indices.any() and bg_indices.any() else "GT semantic fg/bg")
                    ax.set_xlabel("Max cosine sim")
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    fig.suptitle(f"GCLIP Obstacle Sim | ep={episode.episode_id} query={current_obj} result={result.name}")
                    hist_path = out_path.with_suffix(".gclip_sim_hist.png")
                    fig.tight_layout()
                    fig.savefig(str(hist_path), dpi=120)
                    plt.close(fig)
                    print(f"Saved GCLIP sim histogram to: {hist_path}")
        except Exception as e:
            print(f"Warning: could not save GCLIP histogram: {e}")

    if tau_history:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(tau_history, color="steelblue", linewidth=1.5)
            ax.set_xlabel("Step")
            ax.set_ylabel("τ (YOLO-CP threshold)")
            ax.set_title(f"CP threshold τ over episode  |  ep={episode.episode_id}  query={current_obj}  result={result.name}")
            ax.set_ylim(0.0, 1.05)
            ax.axhline(tau_history[0], color="grey", linestyle="--", linewidth=0.8, label=f"init τ={tau_history[0]:.2f}")
            ax.axhline(tau_history[-1], color="coral", linestyle="--", linewidth=0.8, label=f"final τ={tau_history[-1]:.3f}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            tau_path = out_path.with_suffix(".tau_curve.png")
            fig.tight_layout()
            fig.savefig(str(tau_path), dpi=120)
            plt.close(fig)
            print(f"Saved τ curve to: {tau_path}")
        except Exception as e:
            print(f"Warning: could not save τ curve: {e}")

    # Save GT semantic collision map as annotated PNG
    collision_data = evaluator.get_semantic_collision_data(episode.scene_id, current_obj, floor_y=getattr(evaluator, "_episode_floor_y", None))
    map_img = build_ground_truth_semantic_panel(evaluator, episode.scene_id, current_obj)
    scale = 3
    map_img = cv2.resize(map_img, (PANEL_SIZE * scale, PANEL_SIZE * scale), interpolation=cv2.INTER_NEAREST)
    row_h = 32
    legend_h = max(PANEL_SIZE * scale, len(collision_data.labels) * row_h + 20)
    legend_w = 320
    legend = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
    palette = build_palette(collision_data.labels) if collision_data.labels else []
    for i, label in enumerate(collision_data.labels):
        color = tuple(int(c) for c in palette[i])
        y = 20 + i * row_h
        cv2.rectangle(legend, (10, y - 16), (36, y + 8), color, -1)
        cv2.putText(legend, label, (44, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    annotated = np.concatenate([map_img, legend[:map_img.shape[0]]], axis=1)
    map_path = out_path.with_suffix(".semantic_map.png")
    cv2.imwrite(str(map_path), annotated)
    print(f"Saved semantic map to: {map_path}")

    # Save GCLIP argmax label map (per-cell predicted obstacle label)
    mapper = evaluator.actor.mapper
    clip_cp_map = getattr(mapper, "clip_cp_obstacle_map", None)
    use_argmax = getattr(mapper, "use_clip_argmax_obstacle_map", False)
    if clip_cp_map is not None and use_argmax:
        obs_labels = _OBSTACLE_LABELS_CANONICAL  # same order as _SEMANTIC_SAFETY_RADIUS_CELLS
        n_cells = clip_cp_map.n_cells
        argmax_img = np.zeros((n_cells, n_cells, 3), dtype=np.uint8)
        seed_map = clip_cp_map._seed_map  # 0=bg, 1..N=obstacle label idx+1
        nav_np = mapper.one_map.navigable_map.astype(bool) if hasattr(mapper.one_map.navigable_map, 'astype') else np.array(mapper.one_map.navigable_map, dtype=bool)
        argmax_img[nav_np & (seed_map == 0)] = (50, 50, 50)  # explored free → dark gray
        for j, lbl in enumerate(obs_labels):
            seed = (seed_map == j + 1).astype(np.uint8)
            if seed.max() == 0:
                continue
            color = _OBSTACLE_LABEL_COLOR[lbl]
            kernel = clip_cp_map._kernels[j] if j < len(clip_cp_map._kernels) else None
            if kernel is not None:
                dilated = cv2.dilate(seed, kernel, iterations=1).astype(bool)
            else:
                dilated = seed.astype(bool)
            argmax_img[dilated] = color
        argmax_img = cv2.resize(orient_xy_map(argmax_img),
                                (PANEL_SIZE * scale, PANEL_SIZE * scale),
                                interpolation=cv2.INTER_NEAREST)
        # Build legend (only labels that appear in seed_map)
        present_labels = [lbl for j, lbl in enumerate(obs_labels) if (seed_map == j + 1).any()]
        legend2 = np.zeros((legend_h, legend_w, 3), dtype=np.uint8)
        for i, lbl in enumerate(present_labels):
            y_pos = 20 + i * row_h
            color = _OBSTACLE_LABEL_COLOR[lbl]
            cv2.rectangle(legend2, (10, y_pos - 16), (36, y_pos + 8), color, -1)
            cv2.putText(legend2, lbl, (44, y_pos + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        argmax_annotated = np.concatenate([argmax_img, legend2[:argmax_img.shape[0]]], axis=1)
        argmax_path = out_path.with_suffix(".gclip_argmax_map.png")
        cv2.imwrite(str(argmax_path), argmax_annotated)
        print(f"Saved GCLIP argmax label map to: {argmax_path}")

        # Cross-table: for each GT label, what did argmax predict at those cells?
        pred_label_names = ["bg"] + list(obs_labels)
        gt_labels_list = collision_data.labels  # labels[k-1] for gt_label_map value k
        gt_label_map = collision_data.label_map
        bg_winner_map = getattr(clip_cp_map, "_bg_winner_map", None)
        bg_label_names = getattr(clip_cp_map, "_bg_labels", [])
        print("\n--- GT vs GCLIP-argmax cross-table (GT rows, argmax cols) ---")
        header = "GT \\ Pred"
        print(f"{header:<20}", end="")
        for pname in pred_label_names:
            print(f"{pname[:12]:>14}", end="")
        print()
        for gt_idx, gt_lbl in enumerate(gt_labels_list, start=1):
            gt_mask = (gt_label_map == gt_idx)
            if not gt_mask.any():
                continue
            print(f"{gt_lbl:<20}", end="")
            for pred_idx in range(len(pred_label_names)):
                count = int((seed_map[gt_mask] == pred_idx).sum())
                print(f"{count:>14}", end="")
            print(f"  | total={gt_mask.sum()}")
            # Show top-5 winning bg labels at this GT position
            if bg_winner_map is not None and bg_label_names:
                bg_winners = bg_winner_map[gt_mask]
                bg_winners = bg_winners[bg_winners > 0]  # exclude unobserved
                if len(bg_winners) > 0:
                    from collections import Counter as _Counter
                    top = _Counter(bg_winners.tolist()).most_common(5)
                    top_str = ", ".join(
                        f"{bg_label_names[idx-1]}({cnt})"
                        for idx, cnt in top
                        if 0 < idx <= len(bg_label_names)
                    )
                    print(f"  -> top bg winners: {top_str}")
        print("---")

    # CP mode: nonconformity score analysis for GT FG cells
    use_cp = getattr(mapper, "use_clip_cp_obstacle_map", False)
    if clip_cp_map is not None and use_cp and not use_argmax:
        # _last_obs_sims: saved per-cell max obstacle sim (need to add to clip_cp_map)
        # Fallback: use feature_map_gclip to recompute sims for GT FG cells
        gt_label_map = collision_data.label_map
        gt_labels_list = collision_data.labels
        feat_map = getattr(mapper.one_map, "feature_map_gclip", None)
        text_feats = clip_cp_map._text_features  # [N_obs, F]
        if feat_map is not None and text_feats is not None:
            import torch, torch.nn.functional as F
            print("\n--- CP nonconformity score analysis (GT FG cells) ---")
            print("  nonconformity = 1 - max_j sim(cell, obstacle_label_j)")
            feat_np = feat_map  # [n, n, F]
            if hasattr(feat_np, 'cpu'):
                feat_np = feat_np.cpu().numpy()
            n = feat_np.shape[0]
            flat = feat_np.reshape(-1, feat_np.shape[-1])  # [n*n, F]
            flat_t = torch.from_numpy(flat).float()
            norms = flat_t.norm(dim=1, keepdim=True)
            observed = (norms.squeeze() > 1e-6)
            flat_norm = F.normalize(flat_t, dim=1)
            tf = text_feats.cpu().float()
            sims = (flat_norm @ tf.T).numpy()  # [n*n, N_obs]
            max_sims = sims.max(axis=1).reshape(n, n)  # best obstacle sim per cell

            tau_now = clip_cp_map.threshold
            print(f"  Current threshold τ = {tau_now:.4f}")
            for gt_idx, gt_lbl in enumerate(gt_labels_list, start=1):
                gt_mask = (gt_label_map == gt_idx)
                if not gt_mask.any():
                    continue
                scores = max_sims[gt_mask]
                scores = scores[scores > 1e-6]  # observed only
                if len(scores) == 0:
                    continue
                nc_scores = 1.0 - scores  # nonconformity
                covered = (scores >= tau_now).sum()
                print(f"\n  [{gt_lbl}]  n={len(scores)}, covered@τ={covered}/{len(scores)} ({100*covered/len(scores):.0f}%)")
                print(f"    sim: mean={scores.mean():.4f}  std={scores.std():.4f}  min={scores.min():.4f}  max={scores.max():.4f}")
                print(f"    nc:  mean={nc_scores.mean():.4f}  std={nc_scores.std():.4f}")
                for cov in [0.50, 0.75, 0.90, 0.95, 0.99]:
                    tau_needed = float(np.quantile(scores, 1.0 - cov))
                    print(f"    τ for {int(cov*100):2d}% coverage = {tau_needed:.4f}  (nc quantile = {1-tau_needed:.4f})")
            print("---")


if __name__ == "__main__":
    main()
