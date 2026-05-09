"""Merge 5-shard results into a single result directory with correct global episode numbering.

Split assigns episodes by per-scene round-robin: scene j, episode i -> shard (i % N).
Evaluator loads scenes in sorted filename order, so shard s local index k maps to:
    global_idx = (k // eps_per_scene_per_shard) * eps_per_scene + s + (k % eps_per_scene_per_shard) * n_shards

With 11 scenes x 30 ep, 5 shards: eps_per_scene=30, eps_per_scene_per_shard=6.
    global = (k // 6) * 30 + s + (k % 6) * 5

Usage: python scripts/merge_shards.py <run_name> [n_shards]
Example: python scripts/merge_shards.py mp3d_oacp_a70
"""
import os, sys, shutil

N_SCENES = 11
EPS_PER_SCENE = 30
TOTAL_EPS = N_SCENES * EPS_PER_SCENE  # 330


def local_to_global(shard_id, local_idx, n_shards=5):
    eps_per_shard_per_scene = EPS_PER_SCENE // n_shards  # 6
    scene_idx = local_idx // eps_per_shard_per_scene
    within_scene = shard_id + (local_idx % eps_per_shard_per_scene) * n_shards
    return scene_idx * EPS_PER_SCENE + within_scene


def merge(run_name, n_shards=5):
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", run_name)
    eps_per_shard = TOTAL_EPS // n_shards  # 66

    out_state = os.path.join(base, "state")
    out_traj = os.path.join(base, "trajectories")
    out_sim = os.path.join(base, "similarities")
    os.makedirs(out_state, exist_ok=True)
    os.makedirs(out_traj, exist_ok=True)
    os.makedirs(out_sim, exist_ok=True)

    total = 0
    for s in range(n_shards):
        shard_state = os.path.join(base, f"s{s}", "state")
        shard_traj = os.path.join(base, f"s{s}", "trajectories")
        shard_sim = os.path.join(base, f"s{s}", "similarities")

        if not os.path.isdir(shard_state):
            print(f"  shard {s}: no state dir, skipping")
            continue

        copied = 0
        for k in range(eps_per_shard):
            g = local_to_global(s, k, n_shards)

            src = os.path.join(shard_state, f"state_{k}.txt")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_state, f"state_{g}.txt"))
                copied += 1

            src = os.path.join(shard_traj, f"poses_{k}.csv")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_traj, f"poses_{g}.csv"))

            src = os.path.join(shard_sim, f"final_sim_{k}.png")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(out_sim, f"final_sim_{g}.png"))

        print(f"  shard {s}: {copied}/{eps_per_shard} episodes merged")
        total += copied

    print(f"\nTotal: {total}/{TOTAL_EPS} episodes merged into {base}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/merge_shards.py <run_name> [n_shards]")
        sys.exit(1)
    merge(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5)
