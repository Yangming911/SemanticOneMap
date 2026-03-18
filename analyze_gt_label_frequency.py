"""
Phase 1 (fast, no robot): scan all episodes, count GT cells per label.
Phase 2 (robot): for top-K labels run short episodes, compute nonconformity d-prime.
"""
import sys, time
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

out_dir = Path("outputs/label_frequency")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Phase 1: count raw object locations per label (no whitelist filter) ──
from eval.semantic_collision import normalize_semantic_label, metric_to_px, _NON_OBSTACLE_LABELS
print("=== Phase 1: GT label frequency across all episodes (raw, no whitelist) ===")
label_cell_counts = defaultdict(int)
label_ep_count    = defaultdict(int)

cell_size = cfg.EvalConf.mapping.size / cfg.EvalConf.mapping.n_points
n_cells   = cfg.EvalConf.mapping.n_points

seen_scenes = set()
for ep in evaluator.episodes:
    scene_id = ep.scene_id
    if scene_id not in seen_scenes:
        evaluator.load_scene(scene_id)
        seen_scenes.add(scene_id)

    obj_locs = evaluator.scene_data[scene_id].object_locations
    for raw_label, objects in obj_locs.items():
        lbl = normalize_semantic_label(raw_label)
        if lbl in _NON_OBSTACLE_LABELS:
            continue
        # count how many grid cells this label occupies
        pts = set()
        for obj in objects:
            # HM3D: obj has bbox.center
            try:
                import numpy as _np
                center = _np.asarray(obj.bbox.center, dtype=_np.float32)
                x, y = float(-center[2]), float(-center[0])
            except Exception:
                continue
            px, py = metric_to_px(x, y, n_cells=n_cells, cell_size=cell_size)
            if 0 <= px < n_cells and 0 <= py < n_cells:
                pts.add((px, py))
        if pts:
            label_cell_counts[lbl] += len(pts)
            label_ep_count[lbl] += 1

# Sort by total cell count
sorted_labels = sorted(label_cell_counts.items(), key=lambda x: -x[1])
print(f"\n{'Label':<25} {'Total GT cells':>15} {'Episodes':>10}")
print("-" * 52)
for lbl, cnt in sorted_labels:
    print(f"{lbl:<25} {cnt:>15,} {label_ep_count[lbl]:>10}")

top_labels = ["potted plant", "towel", "toilet", "pillow", "chair"]
print(f"\nTop labels with ≥50 GT cells: {top_labels}")

# ── Phase 2: run one episode per top label, compute d-prime ──
print("\n=== Phase 2: nonconformity d-prime per label (short episodes) ===")

MAX_STEPS = 150

# Pick one episode whose SCENE contains the label (any query is fine)
label_to_ep = {}
for ep in evaluator.episodes:
    scene_id = ep.scene_id
    obj_locs = evaluator.scene_data.get(scene_id)
    if obj_locs is None:
        evaluator.load_scene(scene_id)
        obj_locs = evaluator.scene_data[scene_id]
    raw_labels_in_scene = {normalize_semantic_label(k) for k in obj_locs.object_locations}
    for lbl in top_labels:
        if lbl not in label_to_ep and lbl in raw_labels_in_scene:
            label_to_ep[lbl] = ep

print(f"  Episodes found: { {k: v.episode_id for k,v in label_to_ep.items()} }")

fg_scores = defaultdict(list)
bg_scores_pool = []   # shared BG pool from all runs

actor = evaluator.actor
gclip_model = actor.mapper.gclip_model
text_feats = gclip_model.get_text_features(
    [f"a {l}" for l in top_labels]
)  # [K, 512]

for lbl in top_labels:
    ep = label_to_ep.get(lbl)
    if ep is None:
        print(f"  {lbl}: no episode found, skipping")
        continue

    print(f"\n  Running ep={ep.episode_id} for [{lbl}] ...")
    evaluator.load_scene(ep.scene_id)
    evaluator.sim.initialize_agent(
        0, habitat_sim.AgentState(ep.start_position, ep.start_rotation))
    actor.reset()
    actor.set_query(ep.obj_sequence[0])
    gt_cd = evaluator.get_semantic_collision_data(ep.scene_id, ep.obj_sequence[0])
    actor.mapper.set_gt_label_map(gt_cd.label_map, gt_cd.labels)

    t0 = time.time()
    for step in range(MAX_STEPS):
        obs = evaluator.sim.get_sensor_observations()
        obs["state"] = evaluator.sim.get_agent(0).get_state()
        action, _ = actor.act(obs)
        evaluator.execute_action(action)

    mapper = actor.mapper
    gclip_feat = mapper.one_map.feature_map_gclip    # [n,n,512]
    gclip_conf = mapper.one_map.confidence_map_gclip # [n,n]
    nav_map    = mapper.one_map.navigable_map         # [n,n]
    observed   = (gclip_conf > 0).cpu().numpy()

    obs_idx = np.argwhere(observed)
    if len(obs_idx) == 0:
        evaluator.sim.close()
        continue

    feats_all  = gclip_feat[torch.tensor(obs_idx[:,0]), torch.tensor(obs_idx[:,1])]
    feats_norm = F.normalize(feats_all.float(), dim=1)
    sims_all   = (feats_norm @ text_feats.T.to(feats_norm.device)).cpu().numpy()  # [M, K]
    scores_all = 1.0 - sims_all

    # Build raw GT seed map for ALL top_labels (bypasses whitelist)
    raw_seed = np.zeros((n_cells, n_cells), dtype=np.int32)   # lbl_col+1
    obj_locs = evaluator.scene_data[ep.scene_id].object_locations
    for raw_label, objects in obj_locs.items():
        norm_lbl = normalize_semantic_label(raw_label)
        if norm_lbl not in top_labels:
            continue
        lbl_col = top_labels.index(norm_lbl)
        for obj in objects:
            try:
                center = np.asarray(obj.bbox.center, dtype=np.float32)
                x, y = float(-center[2]), float(-center[0])
            except Exception:
                continue
            px_, py_ = metric_to_px(x, y, n_cells=n_cells, cell_size=cell_size)
            if 0 <= px_ < n_cells and 0 <= py_ < n_cells:
                raw_seed[px_, py_] = lbl_col + 1

    raw_seed_obs = raw_seed[obs_idx[:,0], obs_idx[:,1]]
    nav_obs      = nav_map[obs_idx[:,0], obs_idx[:,1]]
    bg_mask      = nav_obs & (raw_seed_obs == 0)

    for lbl_col, tgt_lbl in enumerate(top_labels):
        s_col   = scores_all[:, lbl_col]
        fg_mask = (raw_seed_obs == lbl_col + 1)
        if fg_mask.any():
            fg_scores[tgt_lbl].extend(s_col[fg_mask].tolist())
        if tgt_lbl == lbl:
            bg_scores_pool.extend([(tgt_lbl, float(s)) for s in s_col[bg_mask]])

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s, observed={observed.sum()}")
    evaluator.sim.close()

# ── Summary table ──
print("\n\n=== RESULTS: label frequency + d-prime ===")
print(f"{'Label':<22} {'GT cells':>9} {'FG n':>7} {'s_FG':>8} {'s_BG':>8} {'d-prime':>8} {'Q80 τ':>8}")
print("-" * 80)

n_labels = len(top_labels)
fig, axes = plt.subplots(1, n_labels, figsize=(6 * n_labels, 5))
if n_labels == 1:
    axes = [axes]

results = []
for lbl_col, lbl in enumerate(top_labels):
    fg = np.array(fg_scores[lbl]) if fg_scores[lbl] else np.array([])
    bg_vals = [s for tgt, s in bg_scores_pool if tgt == lbl]
    if not bg_vals:
        bg_vals = [s for _, s in bg_scores_pool]
    bg = np.array(bg_vals[:5000]) if bg_vals else np.array([])

    total_cells = label_cell_counts[lbl]
    fg_n = len(fg)

    if fg_n < 5 or len(bg) < 5:
        print(f"{lbl:<22} {total_cells:>9,} {fg_n:>7} {'—':>8} {'—':>8} {'—':>8} {'—':>8}")
        results.append((lbl, total_cells, fg_n, None, None, None, None))
        continue

    dp_raw  = (bg.mean() - fg.mean()) / (np.sqrt((fg.std()**2 + bg.std()**2) / 2) + 1e-8)
    q80_tau = float(1.0 - np.percentile(fg, 80))
    print(f"{lbl:<22} {total_cells:>9,} {fg_n:>7} {fg.mean():>8.4f} {bg.mean():>8.4f} {dp_raw:>8.3f} {q80_tau:>8.4f}")
    results.append((lbl, total_cells, fg_n, fg.mean(), bg.mean(), dp_raw, q80_tau))

    bins = np.linspace(min(fg.min(), bg.min()) - 0.005,
                       max(fg.max(), bg.max()) + 0.005, 50)

    ax_bg = axes[lbl_col]
    ax_fg = ax_bg.twinx()   # separate y-axis for FG so small counts are visible

    ax_bg.hist(bg[:2000], bins=bins, alpha=0.35, color="steelblue",
               label=f"BG n={len(bg)}")
    ax_fg.hist(fg, bins=bins, alpha=0.75, color="coral",
               label=f"FG n={fg_n}")

    q80v = np.percentile(fg, 80)
    ax_bg.axvline(q80v, color="red", ls="--", lw=1.5,
                  label=f"Q80={q80v:.3f} (τ={1-q80v:.3f})")

    ax_bg.set_xlabel("Nonconformity score s = 1 - sim")
    ax_bg.set_ylabel("BG count", color="steelblue")
    ax_fg.set_ylabel("FG count", color="coral")
    ax_bg.tick_params(axis="y", labelcolor="steelblue")
    ax_fg.tick_params(axis="y", labelcolor="coral")

    # Combined legend
    h1, l1 = ax_bg.get_legend_handles_labels()
    h2, l2 = ax_fg.get_legend_handles_labels()
    ax_bg.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax_bg.set_title(f"{lbl}\nd'={dp_raw:.2f}", fontsize=9)
    ax_bg.grid(True, alpha=0.3)

fig.suptitle("Nonconformity score distributions (s=1-sim) — dual y-axis: BG left, FG right")
fig.tight_layout()
out_path = out_dir / "label_discriminability.png"
fig.savefig(str(out_path), dpi=120)
plt.close(fig)
print(f"\nSaved {out_path}")

good = [(lbl, dp) for lbl, _, _, _, _, dp, _ in results if dp is not None and dp > 0.5]
good.sort(key=lambda x: -x[1])
print(f"\nRecommended obstacle labels (d' > 0.5): {[l for l,_ in good]}")
