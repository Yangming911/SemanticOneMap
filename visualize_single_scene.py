import argparse
import sys
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


PANEL_SIZE = 640
DEFAULT_CONFIG = "config/mon/eval_conf.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent


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
    hsv = np.zeros((len(labels), 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.linspace(0, 179, len(labels), endpoint=False, dtype=np.uint8)
    hsv[:, 0, 1] = 180
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]


@torch.no_grad()
def build_semantic_panel(mapper, labels: List[str], text_features: torch.Tensor) -> np.ndarray:
    feature_map = mapper.one_map.feature_map
    conf_map = mapper.one_map.confidence_map > 0
    n_cells = mapper.one_map.n_cells

    semantic_rgb = np.zeros((n_cells, n_cells, 3), dtype=np.uint8)
    if int(conf_map.sum().item()) == 0:
        return cv2.resize(orient_xy_map(semantic_rgb), (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)

    coords = torch.nonzero(conf_map, as_tuple=False)
    feats = feature_map[conf_map]
    sims = feats @ text_features.T
    label_ids = torch.argmax(sims, dim=1).cpu().numpy()
    coords_np = coords.cpu().numpy()
    palette = build_palette(labels)
    semantic_rgb[coords_np[:, 0], coords_np[:, 1]] = palette[label_ids]

    oriented = orient_xy_map(semantic_rgb)
    return cv2.resize(oriented, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)


def build_obstacle_panel(mapper) -> np.ndarray:
    dilated_obstacles = (~mapper.one_map.navigable_map.astype(bool)).astype(np.uint8)
    panel = np.zeros((dilated_obstacles.shape[0], dilated_obstacles.shape[1], 3), dtype=np.uint8)
    panel[dilated_obstacles > 0] = np.array([230, 230, 230], dtype=np.uint8)
    panel = orient_xy_map(panel)
    return cv2.resize(panel, (PANEL_SIZE, PANEL_SIZE), interpolation=cv2.INTER_NEAREST)


def draw_path_robot_goal(
    panel: np.ndarray,
    mapper,
    robot_px: Tuple[int, int],
    path,
    chosen_detection,
    title: str,
) -> np.ndarray:
    canvas = panel.copy()
    map_shape = mapper.one_map.navigable_map.shape

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


def draw_legend(panel: np.ndarray, labels: List[str]) -> np.ndarray:
    canvas = panel.copy()
    palette = build_palette(labels)
    y = 42
    for idx, label in enumerate(labels):
        if y > canvas.shape[0] - 16:
            break
        color = tuple(int(v) for v in palette[idx].tolist())
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


def resolve_semantic_labels(evaluator: HabitatEvaluator, episode, override: Optional[str]) -> List[str]:
    if override:
        labels = [label.strip() for label in override.split(",") if label.strip()]
        return unique_keep_order(labels)

    scene_labels = sorted(evaluator.scene_data[episode.scene_id].object_locations.keys())
    return unique_keep_order(episode.obj_sequence + scene_labels)


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
    semantic_labels = resolve_semantic_labels(evaluator, episode, args.semantic_labels)
    semantic_text_features = evaluator.actor.mapper.model.get_text_features(
        [f"a {label}" for label in semantic_labels]
    ).to(evaluator.actor.mapper.one_map.map_device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    poses = []
    result = Result.FAILURE_OOT

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
            obstacle_panel = build_obstacle_panel(evaluator.actor.mapper)
            obstacle_panel = draw_path_robot_goal(
                obstacle_panel,
                evaluator.actor.mapper,
                robot_px,
                path,
                chosen_detection,
                "Dilated Obstacles + A*",
            )
            semantic_panel = build_semantic_panel(
                evaluator.actor.mapper,
                semantic_labels,
                semantic_text_features,
            )
            semantic_panel = draw_path_robot_goal(
                semantic_panel,
                evaluator.actor.mapper,
                robot_px,
                path,
                chosen_detection,
                "CLIP Semantic Map",
            )
            semantic_panel = draw_legend(semantic_panel, semantic_labels)

            frame = np.concatenate([rgb_panel, obstacle_panel, semantic_panel], axis=1)

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

            if called_found:
                result = classify_result(evaluator, episode.scene_id, current_obj, True, poses)
                break
        else:
            result = classify_result(evaluator, episode.scene_id, current_obj, False, poses)
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


if __name__ == "__main__":
    main()
