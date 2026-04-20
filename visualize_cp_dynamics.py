"""
Visualize CLIP-CP dynamics over an episode: τ, α, coverage, seed count, etc.

Usage:
  PYTORCH_NO_NVML=1 xvfb-run -a python visualize_cp_dynamics.py \
      --episode-id 8 --no-display \
      -c config/mon/eval_conf_mp3d_wgate_mp3d_v5e_mini.yaml

Produces:
  1. The standard visualization video (same as visualize_single_scene.py)
  2. A multi-panel PNG showing CP metric evolution over time
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import habitat_sim
import numpy as np
import torch

from config import load_eval_config
from eval import get_closest_dist
from eval.actor import MONActor
from eval.habitat_evaluator import HabitatEvaluator, Result
from visualize_single_scene import (
    PANEL_SIZE,
    PROJECT_ROOT,
    build_obstacle_layers_panel,
    build_obstacle_panel,
    build_output_path,
    build_palette,
    build_rgb_panel,
    build_semantic_panel,
    build_ground_truth_semantic_panel,
    build_semantic_candidate_labels,
    build_legend_labels,
    classify_result,
    draw_legend,
    draw_path_robot_goal,
    orient_xy_map,
    px_to_display_xy,
    resolve_eval_paths,
    resolve_semantic_labels,
    select_episode,
    state_to_pose_and_metric_xy,
    unique_keep_order,
)
from vision_models.coco_classes import COCO_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one episode and visualize CP dynamics."
    )
    parser.add_argument("--episode-id", type=int, default=8)
    parser.add_argument("--scene-id", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/cp_dynamics.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--semantic-coco80", action="store_true", default=True)
    parser.add_argument("--no-semantic-coco80", dest="semantic_coco80", action="store_false")
    parser.add_argument("--semantic-threshold", type=float, default=0.0)
    parser.add_argument("--semantic-labels", type=str, default=None)
    return parser.parse_known_args()


def collect_cp_metrics(mapper) -> dict:
    """Snapshot current CLIP-CP state into a metrics dict."""
    cp = getattr(mapper, "clip_cp_obstacle_map", None)
    if cp is None:
        return {}

    n_seeds = int((cp._seed_map > 0).sum())
    n_obstacle = int(cp._obstacle_mask.sum())
    tau = cp.threshold
    alpha = cp._alpha_t
    n_calib = len(cp._scores)

    # Per-label seed counts
    per_label_seeds = {}
    for j, lbl in enumerate(cp._labels):
        per_label_seeds[lbl] = int((cp._seed_map == j + 1).sum())

    # Empirical coverage from calibration buffer
    if n_calib > 0:
        scores = np.array(list(cp._scores))
        C_t = 1.0 - tau  # nonconformity threshold
        emp_coverage = float((scores <= C_t).mean())
    else:
        emp_coverage = float("nan")

    return {
        "tau": tau,
        "alpha": alpha,
        "n_seeds": n_seeds,
        "n_obstacle_cells": n_obstacle,
        "n_calib": n_calib,
        "emp_coverage": emp_coverage,
        "per_label_seeds": per_label_seeds,
    }


def collect_nav_metrics(mapper, evaluator, episode, step) -> dict:
    """Snapshot navigation-relevant metrics."""
    nav_map = mapper.one_map.navigable_map.astype(bool)
    conf_map = mapper.one_map.confidence_map
    explored_cells = int((conf_map > 0).sum()) if hasattr(conf_map, 'sum') else 0
    free_cells = int(nav_map.sum())
    has_path = mapper.path is not None and len(mapper.path) > 1
    path_len = len(mapper.path) if has_path else 0
    has_detection = mapper.chosen_detection is not None

    return {
        "explored_cells": explored_cells,
        "free_cells": free_cells,
        "has_path": has_path,
        "path_len": path_len,
        "has_detection": has_detection,
    }


def plot_cp_dynamics(metrics_history: List[dict], episode, result, out_path: Path):
    """Generate a multi-panel figure showing CP metric evolution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    steps = list(range(len(metrics_history)))
    if len(steps) < 2:
        print("Too few steps for CP dynamics plot.")
        return

    # Extract time series
    tau_series = [m.get("tau", float("nan")) for m in metrics_history]
    alpha_series = [m.get("alpha", float("nan")) for m in metrics_history]
    coverage_series = [m.get("emp_coverage", float("nan")) for m in metrics_history]
    n_seeds_series = [m.get("n_seeds", 0) for m in metrics_history]
    n_obstacle_series = [m.get("n_obstacle_cells", 0) for m in metrics_history]
    n_calib_series = [m.get("n_calib", 0) for m in metrics_history]
    explored_series = [m.get("explored_cells", 0) for m in metrics_history]
    free_series = [m.get("free_cells", 0) for m in metrics_history]
    path_len_series = [m.get("path_len", 0) for m in metrics_history]
    has_detection_series = [1 if m.get("has_detection", False) else 0 for m in metrics_history]

    # Per-label seed counts
    all_labels = set()
    for m in metrics_history:
        all_labels.update(m.get("per_label_seeds", {}).keys())
    label_series = {}
    for lbl in sorted(all_labels):
        label_series[lbl] = [m.get("per_label_seeds", {}).get(lbl, 0) for m in metrics_history]

    # Get target coverage from config
    target_cov = metrics_history[0].get("target_coverage", 0.9)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)

    # --- Panel 1: τ (threshold) over time ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, tau_series, color="steelblue", linewidth=1.5, label="τ")
    ax1.axhline(tau_series[0], color="grey", linestyle="--", linewidth=0.8,
                label=f"init τ={tau_series[0]:.3f}")
    if not np.isnan(tau_series[-1]):
        ax1.axhline(tau_series[-1], color="coral", linestyle="--", linewidth=0.8,
                     label=f"final τ={tau_series[-1]:.3f}")
    ax1.set_ylabel("τ (CP threshold)")
    ax1.set_title("CP Threshold τ")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(steps) - 1)

    # --- Panel 2: α (ACI risk level) over time ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, alpha_series, color="darkorange", linewidth=1.5, label="α_t")
    ax2.axhline(1.0 - target_cov, color="green", linestyle="--", linewidth=0.8,
                label=f"α_target={1.0 - target_cov:.2f}")
    ax2.set_ylabel("α_t (risk level)")
    ax2.set_title("ACI Risk Level α_t")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(steps) - 1)

    # --- Panel 3: Empirical coverage ---
    ax3 = fig.add_subplot(gs[1, 0])
    valid_cov = [(s, c) for s, c in zip(steps, coverage_series) if not np.isnan(c)]
    if valid_cov:
        cov_steps, cov_vals = zip(*valid_cov)
        ax3.plot(cov_steps, cov_vals, color="seagreen", linewidth=1.5, label="emp. coverage")
        ax3.axhline(target_cov, color="red", linestyle="--", linewidth=0.8,
                     label=f"target={target_cov:.2f}")
    ax3.set_ylabel("Coverage")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Empirical Coverage (calibration buffer)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, len(steps) - 1)

    # --- Panel 4: Seed & obstacle cell counts ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, n_seeds_series, color="purple", linewidth=1.2, label="seed cells")
    ax4.plot(steps, n_obstacle_series, color="crimson", linewidth=1.2, label="obstacle cells (dilated)")
    ax4.set_ylabel("Cell count")
    ax4.set_title("CP Obstacle Cells")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, len(steps) - 1)

    # --- Panel 5: Per-label seed breakdown ---
    ax5 = fig.add_subplot(gs[2, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(label_series), 1)))
    for i, (lbl, vals) in enumerate(label_series.items()):
        ax5.plot(steps, vals, linewidth=1.0, label=lbl, color=colors[i % len(colors)])
    ax5.set_ylabel("Seed count")
    ax5.set_title("Per-Label Seed Count")
    ax5.legend(fontsize=7, ncol=2, loc="upper left")
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, len(steps) - 1)

    # --- Panel 6: Calibration buffer size ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(steps, n_calib_series, color="teal", linewidth=1.2, label="calib buffer size")
    ax6.set_ylabel("Count")
    ax6.set_title("Calibration Buffer Size")
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, len(steps) - 1)

    # --- Panel 7: Exploration & navigation ---
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(steps, explored_series, color="steelblue", linewidth=1.0, label="explored cells")
    ax7.plot(steps, free_series, color="green", linewidth=1.0, label="free cells")
    ax7.set_xlabel("Step")
    ax7.set_ylabel("Cell count")
    ax7.set_title("Map Exploration")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, len(steps) - 1)

    # --- Panel 8: Path length & detection status ---
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.plot(steps, path_len_series, color="mediumpurple", linewidth=1.0, label="path length")
    ax8_twin = ax8.twinx()
    ax8_twin.fill_between(steps, has_detection_series, alpha=0.15, color="green", label="has detection")
    ax8_twin.set_ylabel("Detection", color="green")
    ax8_twin.set_ylim(-0.1, 1.5)
    ax8.set_xlabel("Step")
    ax8.set_ylabel("Path length (cells)")
    ax8.set_title("Path & Detection")
    ax8.legend(fontsize=8, loc="upper left")
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, len(steps) - 1)

    fig.suptitle(
        f"CLIP-CP Dynamics | ep={episode.episode_id} query={episode.obj_sequence[0]} "
        f"result={result.name} | {len(steps)} steps",
        fontsize=14,
        fontweight="bold",
    )

    plot_path = out_path.with_suffix(".cp_dynamics.png")
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CP dynamics plot to: {plot_path}")


def plot_calibration_log(calib_log: List[dict], episode, result, out_path: Path):
    """Plot the raw ACI calibration log (per-calibration-event detail)."""
    if not calib_log:
        print("No calibration log entries.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calib_steps = [e["step"] for e in calib_log]
    tau_vals = [e["tau"] for e in calib_log]
    tau_after = [e["tau_after"] for e in calib_log]
    err_vals = [e["err"] for e in calib_log]
    alpha_vals = [e["alpha"] for e in calib_log]
    s_vals = [e["s"] for e in calib_log]

    # Compute rolling coverage (fraction of err==0 in last 50 calibrations)
    window = 50
    rolling_cov = []
    for i in range(len(err_vals)):
        start = max(0, i - window + 1)
        rolling_cov.append(1.0 - np.mean(err_vals[start:i + 1]))

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    # τ per calibration event
    ax = axes[0, 0]
    ax.plot(calib_steps, tau_vals, color="steelblue", linewidth=0.8, alpha=0.7, label="τ (before)")
    ax.plot(calib_steps, tau_after, color="coral", linewidth=0.8, alpha=0.7, label="τ (after)")
    ax.set_ylabel("τ")
    ax.set_title("τ at Each Calibration Event")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # α per calibration event
    ax = axes[0, 1]
    ax.plot(calib_steps, alpha_vals, color="darkorange", linewidth=0.8)
    ax.axhline(0.1, color="green", linestyle="--", linewidth=0.8, label="α_target=0.1")
    ax.set_ylabel("α_t")
    ax.set_title("α_t at Each Calibration")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Nonconformity scores
    ax = axes[1, 0]
    ax.scatter(calib_steps, s_vals, c=err_vals, cmap="RdYlGn_r", s=8, alpha=0.6, edgecolors="none")
    ax.set_ylabel("Nonconformity score s")
    ax.set_title("Nonconformity Scores (red=err, green=covered)")
    ax.grid(True, alpha=0.3)

    # err (binary coverage indicator)
    ax = axes[1, 1]
    ax.bar(calib_steps, err_vals, width=1.0, color="crimson", alpha=0.5, label="err (1=miss)")
    ax.set_ylabel("err")
    ax.set_title("Coverage Errors")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Rolling coverage
    ax = axes[2, 0]
    ax.plot(calib_steps, rolling_cov, color="seagreen", linewidth=1.2, label=f"rolling cov (w={window})")
    ax.axhline(0.9, color="red", linestyle="--", linewidth=0.8, label="target=0.9")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Calibration step")
    ax.set_title(f"Rolling Coverage (window={window})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Histogram of nonconformity scores
    ax = axes[2, 1]
    ax.hist(s_vals, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    if tau_vals:
        final_C = 1.0 - tau_after[-1]
        ax.axvline(final_C, color="red", linestyle="--", label=f"C_t={final_C:.3f}")
    ax.set_xlabel("Nonconformity score s")
    ax.set_title("Score Distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"ACI Calibration Log | ep={episode.episode_id} query={episode.obj_sequence[0]} "
        f"result={result.name} | {len(calib_log)} events",
        fontsize=13,
        fontweight="bold",
    )

    plot_path = out_path.with_suffix(".calib_log.png")
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration log plot to: {plot_path}")


def main():
    args, spock_args = parse_args()
    if not any(arg == "-c" or arg == "--config" for arg in spock_args):
        spock_args = ["-c", "config/mon/eval_conf_mp3d_wgate_mp3d_v5e_mini.yaml"] + spock_args
    sys.argv = [sys.argv[0]] + spock_args

    cfg = load_eval_config()
    resolve_eval_paths(cfg.EvalConf)
    evaluator = HabitatEvaluator(cfg.EvalConf, MONActor(cfg.EvalConf))
    if args.max_steps is not None:
        evaluator.max_steps = args.max_steps

    episode = select_episode(evaluator, args.scene_id, args.episode_id)
    print(f"Episode {episode.episode_id}: scene={episode.scene_id}, query={episode.obj_sequence[0]}")

    evaluator.load_scene(episode.scene_id)
    evaluator.sim.initialize_agent(
        0, habitat_sim.AgentState(episode.start_position, episode.start_rotation)
    )
    evaluator.actor.reset()

    current_obj = episode.obj_sequence[0]
    evaluator.actor.set_query(current_obj)
    evaluator._episode_floor_y = float(episode.start_position[1])

    # Pass GT label map for ACI calibration
    gt_collision = evaluator.get_semantic_collision_data(
        episode.scene_id, current_obj, floor_y=evaluator._episode_floor_y
    )
    evaluator.actor.mapper.set_gt_label_map(gt_collision.label_map, gt_collision.labels)

    # Semantic labels for visualization
    semantic_labels = resolve_semantic_labels(evaluator, episode, args.semantic_labels, args.semantic_coco80)
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
    metrics_history = []  # per-step CP + nav metrics

    # Store target coverage for plotting
    cp_map = getattr(evaluator.actor.mapper, "clip_cp_obstacle_map", None)
    target_coverage = cp_map.target_coverage if cp_map else 0.9

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

            # --- Collect CP metrics ---
            cp_metrics = collect_cp_metrics(evaluator.actor.mapper)
            nav_metrics = collect_nav_metrics(evaluator.actor.mapper, evaluator, episode, step)
            step_metrics = {**cp_metrics, **nav_metrics, "target_coverage": target_coverage}
            metrics_history.append(step_metrics)

            if step % 50 == 0:
                print(
                    f"  step={step:4d}  τ={cp_metrics.get('tau', -1):.4f}  "
                    f"α={cp_metrics.get('alpha', -1):.4f}  "
                    f"cov={cp_metrics.get('emp_coverage', -1):.3f}  "
                    f"seeds={cp_metrics.get('n_seeds', 0)}  "
                    f"obs_cells={cp_metrics.get('n_obstacle_cells', 0)}  "
                    f"calib={cp_metrics.get('n_calib', 0)}"
                )

            # --- Build video frame (reuse visualize_single_scene panels) ---
            active_nav = (
                evaluator.actor.mapper.get_active_navigable_map()
                if hasattr(evaluator.actor.mapper, "get_active_navigable_map")
                else evaluator.actor.mapper.one_map.navigable_map
            )
            map_shape = active_nav.shape

            rgb_panel = build_rgb_panel(
                observations["rgb"],
                [
                    f"ep: {episode.episode_id}  query: {current_obj}",
                    f"step: {step}  tau: {cp_metrics.get('tau', -1):.3f}",
                    f"seeds: {cp_metrics.get('n_seeds', 0)}  cov: {cp_metrics.get('emp_coverage', -1):.2f}",
                ],
            )

            obstacle_panel = build_obstacle_panel(active_nav)
            obstacle_panel = draw_path_robot_goal(
                obstacle_panel, map_shape, robot_px, path, chosen_detection,
                "Active Navigable Map + A*",
            )

            use_yolo = getattr(evaluator.actor.mapper, "use_yolo_obstacle_map", False)
            use_cp = getattr(evaluator.actor.mapper, "use_clip_cp_obstacle_map", False)
            use_clip_cp = (use_cp or getattr(evaluator.actor.mapper, "use_clip_argmax_obstacle_map", False))

            if use_yolo or use_clip_cp:
                layers_panel = build_obstacle_layers_panel(evaluator.actor.mapper)
                layers_panel = draw_path_robot_goal(
                    layers_panel, map_shape, robot_px, path, chosen_detection,
                    "Obstacle Layers",
                )
            else:
                layers_panel = None

            semantic_panel = build_semantic_panel(
                evaluator.actor.mapper, semantic_labels, semantic_text_features,
                args.semantic_threshold, len(semantic_labels),
            )
            semantic_panel = draw_path_robot_goal(
                semantic_panel, map_shape, robot_px, path, chosen_detection,
                "CLIP Semantic Map",
            )
            semantic_panel = draw_legend(
                semantic_panel, semantic_labels, legend_labels=legend_labels, max_items=15,
            )

            gt_panel = build_ground_truth_semantic_panel(evaluator, episode.scene_id, current_obj)
            gt_panel = draw_path_robot_goal(
                gt_panel, map_shape, robot_px, path, chosen_detection,
                "GT Semantic Safety Map",
            )
            gt_panel = draw_legend(gt_panel, gt_collision.labels, max_items=15)

            # Assemble frame
            _half = PANEL_SIZE // 2
            _left_bgr = cv2.cvtColor(observations["rgb_left"][:, :, :3], cv2.COLOR_RGB2BGR)
            _right_bgr = cv2.cvtColor(observations["rgb_right"][:, :, :3], cv2.COLOR_RGB2BGR)
            left_half = cv2.resize(_left_bgr, (PANEL_SIZE, _half), interpolation=cv2.INTER_AREA)
            right_half = cv2.resize(_right_bgr, (PANEL_SIZE, _half), interpolation=cv2.INTER_AREA)
            cv2.putText(left_half, "Left", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(right_half, "Right", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            side_views = np.concatenate([left_half, right_half], axis=0)

            if layers_panel is not None:
                top_row = np.concatenate([side_views, rgb_panel, layers_panel], axis=1)
                bottom_row = np.concatenate(
                    [np.zeros_like(side_views), obstacle_panel, semantic_panel], axis=1
                )
                # Add GT panel - match width
                gt_col = np.concatenate(
                    [np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8), gt_panel], axis=0
                )
                frame = np.concatenate(
                    [np.concatenate([top_row, bottom_row], axis=0)],
                    axis=1,
                )
            else:
                frame = np.concatenate(
                    [side_views, rgb_panel, obstacle_panel, semantic_panel, gt_panel], axis=1
                )

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)

            if not args.no_display:
                cv2.imshow("cp-dynamics", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    result = classify_result(evaluator, episode.scene_id, current_obj, False, poses)
                    break

            evaluator.execute_action(action)

            collided, cause_label = evaluator.check_semantic_collision(episode.scene_id, current_obj)
            if collided:
                result = Result.SEMANTIC_COLLISION
                print(f"SEMANTIC_COLLISION at step={step} cause={cause_label}")
                break

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

    print(f"\nResult: {result.name}")
    print(f"Total steps: {len(metrics_history)}")
    print(f"Video: {out_path}")

    # --- Generate CP dynamics plots ---
    if metrics_history:
        plot_cp_dynamics(metrics_history, episode, result, out_path)

    # --- Generate calibration log plot ---
    cp_map = getattr(evaluator.actor.mapper, "clip_cp_obstacle_map", None)
    if cp_map is not None:
        calib_log = cp_map._calibration_log
        if calib_log:
            plot_calibration_log(calib_log, episode, result, out_path)
            print(f"Total calibration events: {len(calib_log)}")

    if evaluator.sim is not None:
        evaluator.sim.close()


if __name__ == "__main__":
    main()
