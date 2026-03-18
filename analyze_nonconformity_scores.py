"""
Large-scale analysis of GCLIP map nonconformity scores.

For GT obstacle cells vs free-space cells, compute:
  s(j, l_true) = 1 - CosSim(v_j, psi(l_true))   [nonconformity score]

Shows:
  - Distribution of s for GT FG cells vs BG (free) cells per obstacle class
  - Quantile values (80/90/95/99 percentile) for choosing conformal threshold C
  - Separation d-prime in nonconformity space

Runs multiple episodes per obstacle type.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

if "-c" not in sys.argv:
    sys.argv += ["-c", "config/mon/eval_conf_gclip_vis.yaml"]

from config import load_eval_config
from eval.actor import MONActor
from eval.habitat_evaluator import HabitatEvaluator
from visualize_single_scene import resolve_eval_paths
import habitat_sim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = load_eval_config()
resolve_eval_paths(cfg.EvalConf)
evaluator = HabitatEvaluator(cfg.EvalConf, MONActor(cfg.EvalConf))

OBSTACLE_LABELS = ["chair", "potted plant", "toilet"]
MAX_STEPS = 200

# Collect per-class nonconformity scores: fg (GT obstacle) and bg (free)
fg_scores = defaultdict(list)   # label → list of s values for GT cells
bg_scores = defaultdict(list)   # label → list of s values for free cells (using that label's text)

out_dir = Path("outputs/nonconformity_analysis")
out_dir.mkdir(parents=True, exist_ok=True)

# Episodes to run: one per obstacle type to keep it fast
ep_targets = {
    "chair": [6],
    "toilet": [5],
    "potted plant": [1],
}

processed = set()

for target_label, ep_ids in ep_targets.items():
    for ep_id in ep_ids:
        if ep_id in processed:
            continue
        processed.add(ep_id)

        episode = [ep for ep in evaluator.episodes if ep.episode_id == ep_id]
        if not episode:
            continue
        episode = episode[0]

        print(f"\n--- ep={ep_id} query={episode.obj_sequence[0]} ---")
        evaluator.load_scene(episode.scene_id)
        evaluator.sim.initialize_agent(
            0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
        evaluator.actor.reset()
        evaluator.actor.set_query(episode.obj_sequence[0])
        gt_cd = evaluator.get_semantic_collision_data(episode.scene_id, episode.obj_sequence[0])
        evaluator.actor.mapper.set_gt_label_map(gt_cd.label_map, gt_cd.labels)

        # Run episode
        for step in range(MAX_STEPS):
            observations = evaluator.sim.get_sensor_observations()
            observations["state"] = evaluator.sim.get_agent(0).get_state()
            action, _ = evaluator.actor.act(observations)
            evaluator.execute_action(action)

        mapper = evaluator.actor.mapper
        gclip_feat = mapper.one_map.feature_map_gclip    # [n,n,512]
        gclip_conf = mapper.one_map.confidence_map_gclip # [n,n]
        gclip_model = mapper.gclip_model
        nav_map = mapper.one_map.navigable_map            # [n,n] bool

        text_feats = gclip_model.get_text_features(
            [f"a {l}" for l in OBSTACLE_LABELS]
        )  # [3, 512], on CUDA

        observed = (gclip_conf > 0).cpu().numpy()
        label_map = gt_cd.label_map
        gt_labels = gt_cd.labels

        # Get all observed features and compute nonconformity scores vs each label
        obs_idx = np.argwhere(observed)  # [M, 2]
        if len(obs_idx) == 0:
            evaluator.sim.close()
            continue

        feats_all = gclip_feat[torch.tensor(obs_idx[:, 0]), torch.tensor(obs_idx[:, 1])]  # [M, 512]
        feats_norm = F.normalize(feats_all.float(), dim=1)
        sims_all = (feats_norm @ text_feats.T.to(feats_norm.device)).cpu().numpy()  # [M, 3]
        scores_all = 1.0 - sims_all  # nonconformity scores [M, 3]

        # Build masks per cell
        lbl_map_obs = label_map[obs_idx[:, 0], obs_idx[:, 1]]  # [M]
        nav_obs = nav_map[obs_idx[:, 0], obs_idx[:, 1]]         # [M] bool

        for lbl_col, lbl in enumerate(OBSTACLE_LABELS):
            s_col = scores_all[:, lbl_col]  # [M] nonconformity scores for this label

            # FG: cells whose GT label matches this obstacle
            gt_lbl_idx = None
            for k, gl in enumerate(gt_labels):
                if gl == lbl:
                    gt_lbl_idx = k + 1
                    break
            if gt_lbl_idx is not None:
                fg_mask = (lbl_map_obs == gt_lbl_idx)
                if fg_mask.any():
                    fg_scores[lbl].extend(s_col[fg_mask].tolist())
                    print(f"  {lbl}: {fg_mask.sum()} GT cells, "
                          f"s_mean={s_col[fg_mask].mean():.4f}±{s_col[fg_mask].std():.4f}")

            # BG: navigable (free-space) cells
            bg_mask = nav_obs & (lbl_map_obs == 0)
            if bg_mask.any():
                bg_scores[lbl].extend(s_col[bg_mask].tolist())

        evaluator.sim.close()

# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n\n=== NONCONFORMITY SCORE ANALYSIS ===")
print("(Lower s = better match. Threshold C: label in prediction set if s ≤ C)")
print()

fig, axes = plt.subplots(1, len(OBSTACLE_LABELS), figsize=(18, 5))

for col, lbl in enumerate(OBSTACLE_LABELS):
    fg = np.array(fg_scores[lbl]) if fg_scores[lbl] else np.array([])
    bg = np.array(bg_scores[lbl]) if bg_scores[lbl] else np.array([])

    print(f"--- {lbl} ---")
    if len(fg) == 0:
        print("  No GT cells observed. Skipping.")
        continue

    print(f"  FG (GT obstacle): n={len(fg)}, mean={fg.mean():.4f}, std={fg.std():.4f}")
    for q in [50, 80, 90, 95, 99]:
        print(f"    Q{q}={np.percentile(fg, q):.4f}", end="")
    print()

    if len(bg) > 0:
        print(f"  BG (free space):  n={len(bg)}, mean={bg.mean():.4f}, std={bg.std():.4f}")
        pooled = np.sqrt((fg.std()**2 + bg.std()**2) / 2)
        dp = (bg.mean() - fg.mean()) / (pooled + 1e-8)  # BG - FG (should be positive if FG < BG)
        print(f"  d-prime (nonconf, BG-FG): {dp:.4f}")
        print(f"  Suggested C = Q90(FG): {np.percentile(fg, 90):.4f}  "
              f"→ τ = 1-C = {1-np.percentile(fg, 90):.4f}")
        print(f"  Suggested C = Q95(FG): {np.percentile(fg, 95):.4f}  "
              f"→ τ = 1-C = {1-np.percentile(fg, 95):.4f}")

    # Plot
    ax = axes[col]
    all_scores = np.concatenate([fg, bg[:min(len(bg), 5000)]])
    bins = np.linspace(all_scores.min() - 0.01, all_scores.max() + 0.01, 60)
    if len(bg) > 0:
        ax.hist(bg[:5000], bins=bins, alpha=0.5, color="steelblue", label=f"BG free n={min(len(bg),5000)}")
    ax.hist(fg, bins=bins, alpha=0.7, color="coral", label=f"FG GT n={len(fg)}")
    for q, color, ls in [(90, "red", "--"), (95, "darkred", ":")]:
        qv = np.percentile(fg, q)
        ax.axvline(qv, color=color, linestyle=ls, label=f"Q{q}(FG)={qv:.3f}")
    ax.set_title(f"{lbl}\ns=1-CosSim(feat, text)")
    ax.set_xlabel("Nonconformity score s")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle("Nonconformity Score Distributions: GT Obstacle vs Free Space")
fig.tight_layout()
out_path = out_dir / "nonconformity_distributions.png"
fig.savefig(str(out_path), dpi=120)
plt.close(fig)
print(f"\nSaved {out_path}")
