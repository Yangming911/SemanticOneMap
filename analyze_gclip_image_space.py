"""Analyze GCLIP per-pixel similarity in IMAGE space (no map projection).

Grabs N frames from a Habitat episode, runs GCLIP, computes per-pixel cosine sim
to obstacle labels, and visualizes heatmaps + histograms.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import habitat_sim
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from config import load_eval_config
from eval.actor import MONActor
from eval.habitat_evaluator import HabitatEvaluator
from vision_models.gclip_dense import GCLIPModel

OBSTACLE_LABELS = ["chair", "potted plant", "toilet"]
SAMPLE_STEPS = [72, 100, 150, 200, 300]  # steps to sample (after 72 init turns)
PANEL = 640


def main():
    # Use same config as visualization
    config_path = "config/mon/eval_conf_gclip_vis.yaml"
    if "-c" not in sys.argv:
        sys.argv += ["-c", config_path]

    cfg = load_eval_config()
    # resolve paths
    from visualize_single_scene import resolve_eval_paths
    resolve_eval_paths(cfg.EvalConf)

    evaluator = HabitatEvaluator(cfg.EvalConf, MONActor(cfg.EvalConf))

    ep_id = 5
    episode = [ep for ep in evaluator.episodes if ep.episode_id == ep_id][0]
    evaluator.load_scene(episode.scene_id)
    evaluator.sim.initialize_agent(
        0, habitat_sim.AgentState(episode.start_position, episode.start_rotation)
    )
    evaluator.actor.reset()
    evaluator.actor.set_query(episode.obj_sequence[0])

    # Load GCLIP model
    gclip = evaluator.actor.mapper.gclip_model
    if gclip is None:
        print("GCLIP model not loaded! Check config.")
        return

    # Precompute text features
    text_feats = gclip.get_text_features(
        [f"a {l}" for l in OBSTACLE_LABELS]
    )  # [3, 512]

    out_dir = Path("outputs/gclip_image_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Sanity check: run on smoke-test image first ──────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    smoke_img_path = "vision_models/rgb.jpg"
    smoke_img = cv2.imread(smoke_img_path)
    if smoke_img is not None:
        smoke_rgb = cv2.cvtColor(smoke_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)  # [C,H,W]
        with torch.no_grad():
            smoke_feats = gclip.get_image_features(smoke_rgb).squeeze(0)  # [512,40,40]
        smoke_feats_hw = smoke_feats.permute(1, 2, 0).reshape(-1, 512)
        smoke_norm = F.normalize(smoke_feats_hw, dim=1)
        smoke_sims = (smoke_norm @ text_feats.T).reshape(40, 40, len(OBSTACLE_LABELS))
        smoke_max = smoke_sims.max(dim=2)[0].cpu().numpy()
        print(f"[Smoke test rgb.jpg] sim range: [{smoke_max.min():.4f}, {smoke_max.max():.4f}] "
              f"mean={smoke_max.mean():.4f} max-mean={smoke_max.max()-smoke_max.mean():.4f}")
        fig, axes = plt.subplots(1, 2+len(OBSTACLE_LABELS), figsize=(20, 4))
        axes[0].imshow(cv2.cvtColor(smoke_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Smoke test image (rgb.jpg)")
        axes[0].axis("off")
        # auto-scale
        v0, v1 = smoke_max.min()-0.002, smoke_max.max()+0.002
        im = axes[1].imshow(smoke_max, cmap="jet", vmin=v0, vmax=v1)
        axes[1].set_title(f"Max sim AUTO [{v0:.3f},{v1:.3f}]")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        for j, lbl in enumerate(OBSTACLE_LABELS):
            sim_j = smoke_sims[:,:,j].cpu().numpy()
            v0j, v1j = sim_j.min()-0.002, sim_j.max()+0.002
            im = axes[2+j].imshow(sim_j, cmap="jet", vmin=v0j, vmax=v1j)
            axes[2+j].set_title(f'sim("{lbl}") [{sim_j.min():.3f},{sim_j.max():.3f}]')
            axes[2+j].axis("off")
            plt.colorbar(im, ax=axes[2+j], fraction=0.046)
        fig.tight_layout()
        fig.savefig(str(out_dir / "sanity_smoke_test.png"), dpi=120)
        plt.close(fig)
        print(f"  Saved sanity_smoke_test.png")
    else:
        print(f"[Warning] Could not load {smoke_img_path}, skipping sanity check")

    collected = []

    print(f"Running episode {ep_id}, collecting frames at steps {SAMPLE_STEPS}...")
    max_step = max(SAMPLE_STEPS) + 1
    for step in range(max_step):
        observations = evaluator.sim.get_sensor_observations()
        observations["state"] = evaluator.sim.get_agent(0).get_state()

        if step in SAMPLE_STEPS:
            rgb = observations["rgb"][:, :, :3]  # [H, W, 3] uint8
            depth = observations["depth"]

            # Run GCLIP
            # Model expects [1, 3, H, W] float
            img_np = rgb.transpose(2, 0, 1)  # [3, H, W]
            with torch.no_grad():
                gclip_feats = gclip.get_image_features(
                    img_np[np.newaxis, ...]
                ).squeeze(0)  # [512, 40, 40]

            # Per-pixel cosine sim at 40x40 resolution
            feats_hw = gclip_feats.permute(1, 2, 0)  # [40, 40, 512]
            feats_norm = F.normalize(feats_hw.reshape(-1, 512), dim=1)  # [1600, 512]
            text_norm = text_feats.to(feats_norm.device)
            sims = feats_norm @ text_norm.T  # [1600, 3]
            sims = sims.reshape(40, 40, len(OBSTACLE_LABELS))

            max_sim, max_idx = sims.max(dim=2)  # [40, 40]
            per_label_sims = {
                lbl: sims[:, :, j].cpu().numpy()
                for j, lbl in enumerate(OBSTACLE_LABELS)
            }
            max_sim_np = max_sim.cpu().numpy()

            # Downsample depth to 40x40 for comparison
            depth_40 = cv2.resize(depth, (40, 40), interpolation=cv2.INTER_AREA)

            collected.append({
                "step": step,
                "rgb": rgb,
                "depth_40": depth_40,
                "max_sim": max_sim_np,
                "per_label": per_label_sims,
                "max_idx": max_idx.cpu().numpy(),
            })
            print(f"  Step {step}: max_sim range [{max_sim_np.min():.3f}, {max_sim_np.max():.3f}], "
                  f"mean={max_sim_np.mean():.3f}")

        # Execute action
        action, _ = evaluator.actor.act(observations)
        evaluator.execute_action(action)

    evaluator.sim.close()

    # Analyze and plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for frame in collected:
        step = frame["step"]
        rgb = frame["rgb"]
        max_sim = frame["max_sim"]
        depth_40 = frame["depth_40"]
        per_label = frame["per_label"]

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # Row 1: RGB, max sim heatmap, depth
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title(f"RGB (step {step})")
        axes[0, 0].axis("off")

        # Auto-scale: use tight range around actual data to reveal spatial variation
        vmin_auto = max_sim.min() - 0.005
        vmax_auto = max_sim.max() + 0.005
        im = axes[0, 1].imshow(max_sim, cmap="jet", vmin=vmin_auto, vmax=vmax_auto)
        axes[0, 1].set_title(f"Max obstacle sim (40x40) AUTO-SCALE\n"
                             f"range=[{max_sim.min():.3f},{max_sim.max():.3f}] mean={max_sim.mean():.3f}")
        axes[0, 1].axis("off")
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

        axes[0, 2].imshow(depth_40, cmap="gray")
        axes[0, 2].set_title("Depth (40x40)")
        axes[0, 2].axis("off")

        # Row 2: per-label sim heatmaps (auto-scale per label)
        for j, lbl in enumerate(OBSTACLE_LABELS):
            sim_j = per_label[lbl]
            v0, v1 = sim_j.min() - 0.002, sim_j.max() + 0.002
            im = axes[1, j].imshow(sim_j, cmap="jet", vmin=v0, vmax=v1)
            axes[1, j].set_title(f'sim("{lbl}") AUTO-SCALE\nmin={sim_j.min():.3f} max={sim_j.max():.3f}')
            axes[1, j].axis("off")
            plt.colorbar(im, ax=axes[1, j], fraction=0.046)

        fig.suptitle(f"GCLIP Image-Space Analysis | ep={ep_id} step={step}", fontsize=14)
        fig.tight_layout()
        path = out_dir / f"gclip_image_step{step:04d}.png"
        fig.savefig(str(path), dpi=120)
        plt.close(fig)
        print(f"Saved {path}")

    # Aggregate histogram across all frames
    all_max_sims = np.concatenate([f["max_sim"].flatten() for f in collected])
    # Use depth as rough fg/bg proxy: close objects (depth < 2m) vs far (depth > 3m)
    all_depths = np.concatenate([f["depth_40"].flatten() for f in collected])
    near = all_depths < 2.0
    far = (all_depths > 3.0) & (all_depths < 10.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(-0.1, 0.5, 80)

    ax = axes[0]
    ax.hist(all_max_sims, bins=bins, alpha=0.7, color="steelblue")
    ax.set_title(f"All pixels (n={len(all_max_sims)})")
    ax.set_xlabel("Max cosine sim to obstacle labels")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if near.any():
        ax.hist(all_max_sims[near], bins=bins, alpha=0.6,
                label=f"Near <2m (n={near.sum()})", color="coral")
    if far.any():
        ax.hist(all_max_sims[far], bins=bins, alpha=0.6,
                label=f"Far >3m (n={far.sum()})", color="steelblue")
    if near.any() and far.any():
        near_mean = all_max_sims[near].mean()
        far_mean = all_max_sims[far].mean()
        pooled = np.sqrt((all_max_sims[near].std()**2 + all_max_sims[far].std()**2) / 2)
        dp = (near_mean - far_mean) / pooled if pooled > 1e-8 else 0
        ax.set_title(f"Near vs Far | d'={dp:.2f}")
    ax.set_xlabel("Max cosine sim")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"GCLIP Image-Space Aggregate | ep={ep_id}")
    fig.tight_layout()
    hist_path = out_dir / "gclip_image_aggregate_hist.png"
    fig.savefig(str(hist_path), dpi=120)
    plt.close(fig)
    print(f"Saved {hist_path}")

    # Per-label aggregate stats
    print("\n=== Per-label image-space stats (all frames) ===")
    for lbl in OBSTACLE_LABELS:
        all_lbl = np.concatenate([f["per_label"][lbl].flatten() for f in collected])
        print(f"  {lbl}: mean={all_lbl.mean():.4f}, std={all_lbl.std():.4f}, "
              f"min={all_lbl.min():.4f}, max={all_lbl.max():.4f}")
        if near.any():
            print(f"    near<2m: mean={all_lbl[near].mean():.4f}±{all_lbl[near].std():.4f}")
        if far.any():
            print(f"    far>3m:  mean={all_lbl[far].mean():.4f}±{all_lbl[far].std():.4f}")


if __name__ == "__main__":
    main()
