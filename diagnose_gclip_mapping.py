"""
Diagnostic: trace GCLIP sim from image space → 2D map, identify where signal is lost.

For a single frame (step 300, chair visible):
  1. Image space: GCLIP 40×40 sim to "a chair"
  2. Per-patch depth center → which grid cell does it land on?
  3. Color the 2D grid cells by the GCLIP sim of their contributing patch
  4. Overlay with GT semantic collision map to check alignment
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import habitat_sim
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ── project helpers copied from feature_map.py ─────────────────────────────
def rotate_pcl_np(pcl: np.ndarray, tf: np.ndarray) -> np.ndarray:
    """pcl: [N,3], tf: [4,4] → rotated [N,3]"""
    ones = np.ones((pcl.shape[0], 1))
    pcl4 = np.concatenate([pcl, ones], axis=1)   # [N,4]
    out = (tf @ pcl4.T).T                          # [N,4]
    return out[:, :3]


def back_project(depth_hw: np.ndarray, fx, fy, cx, cy) -> np.ndarray:
    """depth [H,W] → point cloud [H*W, 3] in camera frame (x=fwd,y=left,z=up)"""
    H, W = depth_hw.shape
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    zz = depth_hw
    x_w = (xx - cx) * zz / fx
    y_w = (yy - cy) * zz / fy
    z_w = zz
    # camera convention used in feature_map.py: (z_world, -x_world, -y_world)
    pcl = np.stack([z_w, -x_w, -y_w], axis=-1).reshape(-1, 3)
    return pcl


def pcl_to_grid(pcl_world: np.ndarray, n_cells: int, cell_size: float,
                map_center_cells: np.ndarray) -> np.ndarray:
    """[N,3] world pcl → grid ids [N,2] (may be out of bounds)"""
    gx = np.floor(pcl_world[:, 0] / cell_size).astype(np.int32) + map_center_cells[0]
    gy = np.floor(pcl_world[:, 1] / cell_size).astype(np.int32) + map_center_cells[1]
    return np.stack([gx, gy], axis=1)


# ── main ────────────────────────────────────────────────────────────────────
def main():
    TARGET_STEP = 300

    if "-c" not in sys.argv:
        sys.argv += ["-c", "config/mon/eval_conf_gclip_vis.yaml"]

    from config import load_eval_config
    from eval.actor import MONActor
    from eval.habitat_evaluator import HabitatEvaluator
    from visualize_single_scene import resolve_eval_paths
    import torch.nn.functional as F

    cfg = load_eval_config()
    resolve_eval_paths(cfg.EvalConf)
    evaluator = HabitatEvaluator(cfg.EvalConf, MONActor(cfg.EvalConf))

    ep_id = 5
    episode = [ep for ep in evaluator.episodes if ep.episode_id == ep_id][0]
    evaluator.load_scene(episode.scene_id)
    evaluator.sim.initialize_agent(
        0, habitat_sim.AgentState(episode.start_position, episode.start_rotation))
    evaluator.actor.reset()
    evaluator.actor.set_query(episode.obj_sequence[0])

    mapper = evaluator.actor.mapper
    gclip = mapper.gclip_model
    one_map = mapper.one_map

    out_dir = Path("outputs/gclip_mapping_diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run until target step
    for step in range(TARGET_STEP + 1):
        observations = evaluator.sim.get_sensor_observations()
        observations["state"] = evaluator.sim.get_agent(0).get_state()
        action, _ = evaluator.actor.act(observations)

        if step == TARGET_STEP:
            rgb = observations["rgb"][:, :, :3]        # [H,W,3] uint8 RGB
            depth = observations["depth"].squeeze()     # [H,W] float32
            state = observations["state"]
            break
        evaluator.execute_action(action)

    # ── 1. GCLIP features in image space ────────────────────────────────────
    img_chw = rgb.transpose(2, 0, 1)   # [C,H,W]
    with torch.no_grad():
        gclip_feats = gclip.get_image_features(img_chw).squeeze(0)  # [512,40,40]

    text_feats = gclip.get_text_features(["a chair", "a potted plant", "a toilet"])
    feats_40 = gclip_feats.permute(1, 2, 0)   # [40,40,512]
    feats_norm = F.normalize(feats_40.reshape(-1, 512), dim=1)
    sims_40 = (feats_norm @ text_feats.T).reshape(40, 40, 3)  # [40,40,3]
    chair_sim_40 = sims_40[:, :, 0].cpu().numpy()              # [40,40]

    # ── 2. Per-patch: depth center + grid projection ─────────────────────────
    H, W = depth.shape
    fx = one_map.fx; fy = one_map.fy; cx = one_map.cx; cy = one_map.cy
    n_cells = one_map.n_cells
    cell_size = one_map.cell_size
    map_center = one_map.map_center_cells.cpu().numpy().astype(np.int32)

    # Build odometry exactly as actor.py does
    from scipy.spatial.transform import Rotation
    pos_arr = np.array([[-state.position[2]], [-state.position[0]], [state.position[1]]])
    orientation = state.rotation
    r = Rotation.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
    yaw, _, _ = r.as_euler("yxz")
    r_mat = Rotation.from_euler("xyz", [0, 0, yaw]).as_matrix()
    tf = np.vstack([np.hstack([r_mat, pos_arr]), [0, 0, 0, 1]])  # [4,4]

    # Project full depth → 3D → world
    pcl_cam = back_project(depth, fx, fy, cx, cy)   # [H*W, 3]
    pcl_world = rotate_pcl_np(pcl_cam, tf)

    cam_x = tf[0, 3]
    cam_y = tf[1, 3]
    pcl_world[:, 0] += cam_x
    pcl_world[:, 1] += cam_y

    grid_ids = pcl_to_grid(pcl_world, n_cells, cell_size, map_center)   # [H*W, 2]

    # ── 3. For each of the 40×40 patches, find the grid cells it contributes to
    # Subsample: for each patch (r,c) in 40×40, use the center pixel
    patch_h = H // 40
    patch_w = W // 40

    # Build sim map in 2D grid space using patch-to-cell projection
    # (project ONLY patch-center pixels, no bilinear interpolation)
    grid_sim_map = np.full((n_cells, n_cells), np.nan, dtype=np.float32)
    grid_count = np.zeros((n_cells, n_cells), dtype=np.int32)

    patch_positions = []  # (grid_r, grid_c, sim_val)

    for pr in range(40):
        for pc in range(40):
            # Center pixel of this patch
            py = int((pr + 0.5) * patch_h)
            px = int((pc + 0.5) * patch_w)
            flat_idx = py * W + px
            d = depth[py, px]
            if d == 0 or d == float('inf') or d > 15:
                continue
            gx, gy = grid_ids[flat_idx]
            if 0 <= gx < n_cells and 0 <= gy < n_cells:
                sim_val = float(chair_sim_40[pr, pc])
                if np.isnan(grid_sim_map[gx, gy]):
                    grid_sim_map[gx, gy] = sim_val
                else:
                    grid_sim_map[gx, gy] = (grid_sim_map[gx, gy] + sim_val) / 2
                grid_count[gx, gy] += 1
                patch_positions.append((gx, gy, sim_val))

    print(f"Patches projected: {len(patch_positions)}")
    valid_sims = grid_sim_map[~np.isnan(grid_sim_map)]
    if len(valid_sims) > 0:
        print(f"Grid sim range: [{valid_sims.min():.4f}, {valid_sims.max():.4f}] "
              f"mean={valid_sims.mean():.4f}")

    # ── 4. GT semantic collision map ─────────────────────────────────────────
    gt_cd = evaluator.get_semantic_collision_data(episode.scene_id, episode.obj_sequence[0])
    gt_map = gt_cd.label_map  # [n,n], 0=bg, 1+=obstacle

    # ── 5. Plot ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 10))

    # ── Row 1: image space ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(rgb)
    ax1.set_title(f"RGB (step {TARGET_STEP})")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 4, 2)
    v0, v1 = chair_sim_40.min()-0.002, chair_sim_40.max()+0.002
    im2 = ax2.imshow(chair_sim_40, cmap="jet", vmin=v0, vmax=v1)
    ax2.set_title(f'Image sim("chair") 40×40\n[{v0:.3f},{v1:.3f}] max-mean={chair_sim_40.max()-chair_sim_40.mean():.4f}')
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 4, 3)
    dep_disp = depth.copy()
    dep_disp[dep_disp > 10] = 10
    ax3.imshow(dep_disp, cmap="gray")
    ax3.set_title("Depth")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 4, 4)
    nav = one_map.navigable_map.astype(np.float32)
    ax4.imshow(nav.T, cmap="gray", origin="lower")
    ax4.set_title("Navigable map (white=free)")
    ax4.axis("off")

    # ── Row 2: grid space ────────────────────────────────────────────────────
    # Crop to explored region for clarity
    has_feat = ~np.isnan(grid_sim_map)
    if has_feat.any():
        rows = np.where(has_feat.any(axis=1))[0]
        cols = np.where(has_feat.any(axis=0))[0]
        r0, r1 = max(0, rows[0]-5), min(n_cells, rows[-1]+5)
        c0, c1 = max(0, cols[0]-5), min(n_cells, cols[-1]+5)
    else:
        r0, r1, c0, c1 = 0, n_cells, 0, n_cells

    crop = lambda m: m[r0:r1, c0:c1]

    ax5 = fig.add_subplot(2, 4, 5)
    gsm_crop = crop(grid_sim_map)
    valid = ~np.isnan(gsm_crop)
    if valid.any():
        vv0 = gsm_crop[valid].min()-0.002
        vv1 = gsm_crop[valid].max()+0.002
    else:
        vv0, vv1 = 0.2, 0.3
    display = np.where(np.isnan(gsm_crop), vv0-0.01, gsm_crop)
    im5 = ax5.imshow(display.T, cmap="jet", vmin=vv0, vmax=vv1, origin="lower")
    ax5.set_title(f'Grid sim("chair") per-patch center\n[{vv0:.3f},{vv1:.3f}]')
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046)

    ax6 = fig.add_subplot(2, 4, 6)
    gt_crop = crop(gt_map)
    ax6.imshow(gt_crop.T, cmap="tab10", vmin=0, vmax=5, origin="lower")
    ax6.set_title(f"GT semantic map\nlabels={gt_cd.labels}")
    ax6.axis("off")

    ax7 = fig.add_subplot(2, 4, 7)
    # Overlay: GT chair cells on grid sim map
    overlay = np.zeros((*gsm_crop.shape, 3), dtype=np.float32)
    if valid.any():
        norm_sim = (gsm_crop - vv0) / (vv1 - vv0 + 1e-8)
        norm_sim = np.clip(norm_sim, 0, 1)
        cmap = plt.cm.jet
        overlay = cmap(norm_sim)[:, :, :3]
    overlay[np.isnan(gsm_crop)] = [0.1, 0.1, 0.1]
    # Mark GT chair cells with white border
    if len(gt_cd.labels) > 0:
        for lbl_idx, lbl in enumerate(gt_cd.labels):
            lbl_mask = (gt_crop == lbl_idx + 1)
            overlay[lbl_mask, :] = [1.0, 1.0, 1.0]  # white = GT obstacle
    ax7.imshow(overlay.transpose(1,0,2), origin="lower")
    ax7.set_title("Grid sim + GT overlay\n(white = GT obstacle cell)")
    ax7.axis("off")

    # Stats: sim at GT cells vs non-GT cells
    gt_mask_full = gt_map > 0
    has_sim = ~np.isnan(grid_sim_map)
    fg = has_sim & gt_mask_full
    bg = has_sim & ~gt_mask_full & ~one_map.navigable_map  # non-GT obstacles

    ax8 = fig.add_subplot(2, 4, 8)
    bins = np.linspace(valid_sims.min()-0.01, valid_sims.max()+0.01, 40)
    if bg.any():
        ax8.hist(grid_sim_map[bg], bins=bins, alpha=0.6,
                 label=f"non-GT obstacle (n={bg.sum()})", color="steelblue")
    if fg.any():
        ax8.hist(grid_sim_map[fg], bins=bins, alpha=0.6,
                 label=f"GT obstacle (n={fg.sum()})", color="coral")
    if fg.any() and bg.any():
        dp_pooled = np.sqrt((grid_sim_map[fg].std()**2 + grid_sim_map[bg].std()**2)/2)
        dp = (grid_sim_map[fg].mean() - grid_sim_map[bg].mean()) / (dp_pooled + 1e-8)
        ax8.set_title(f"Grid cell sim histogram\nd'={dp:.3f}")
    else:
        ax8.set_title("Grid cell sim histogram")
    ax8.legend(fontsize=8)
    ax8.set_xlabel("sim(\"chair\")")
    ax8.grid(True, alpha=0.3)

    fig.suptitle(f"GCLIP Mapping Diagnostic | ep={ep_id} step={TARGET_STEP} query={episode.obj_sequence[0]}")
    fig.tight_layout()
    out_path = out_dir / f"diag_step{TARGET_STEP:04d}.png"
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")

    if fg.any() and bg.any():
        print(f"\n=== Grid-space d-prime (patch-center projection, no blur) ===")
        print(f"  GT_FG: n={fg.sum()}, mean={grid_sim_map[fg].mean():.4f}±{grid_sim_map[fg].std():.4f}")
        print(f"  non-GT obstacle: n={bg.sum()}, mean={grid_sim_map[bg].mean():.4f}±{grid_sim_map[bg].std():.4f}")
        print(f"  d-prime = {dp:.4f}")

    evaluator.sim.close()


if __name__ == "__main__":
    main()
