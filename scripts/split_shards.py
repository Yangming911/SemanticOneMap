"""Split val_ablation dataset into N shards (reverse of merge_shards.py).

Each scene has 30 episodes. Episode i within a scene goes to shard (i % n_shards).
This matches the merge_shards.py local_to_global mapping:
    shard s, local_idx k -> global = (k // eps_per_shard_per_scene) * 30 + s + (k % eps_per_shard_per_scene) * n_shards

Usage: python scripts/split_shards.py [n_shards]
Default: 5 shards (val_ablation_s0 .. val_ablation_s4)
"""
import os, sys, gzip, json, shutil

EPS_PER_SCENE = 30
SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "objectnav_mp3d_v1", "val_ablation", "content",
)
DST_BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "objectnav_mp3d_v1",
)


def split(n_shards=5):
    eps_per_shard = EPS_PER_SCENE // n_shards

    dst_dirs = []
    for s in range(n_shards):
        d = os.path.join(DST_BASE, f"val_ablation_s{s}", "content")
        os.makedirs(d, exist_ok=True)
        dst_dirs.append(d)

    scene_files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".json.gz"))
    print(f"Splitting {len(scene_files)} scenes into {n_shards} shards "
          f"({eps_per_shard} episodes/scene/shard)")

    for scene_file in scene_files:
        src_path = os.path.join(SRC_DIR, scene_file)
        with gzip.open(src_path, "rt") as f:
            data = json.load(f)

        episodes = data["episodes"]
        assert len(episodes) == EPS_PER_SCENE, \
            f"{scene_file}: expected {EPS_PER_SCENE} episodes, got {len(episodes)}"

        meta = {k: v for k, v in data.items() if k != "episodes"}

        for s in range(n_shards):
            shard_episodes = [ep for i, ep in enumerate(episodes) if i % n_shards == s]

            for local_idx, ep in enumerate(shard_episodes):
                ep["episode_id"] = str(local_idx)

            shard_data = {**meta, "episodes": shard_episodes}

            dst_path = os.path.join(dst_dirs[s], scene_file)
            with gzip.open(dst_path, "wt") as f:
                json.dump(shard_data, f)

        print(f"  {scene_file}: {len(episodes)} -> {n_shards} x {eps_per_shard}")

    # Remove extra shard dirs (e.g., s5 if switching from 6 to 5 shards)
    for s in range(n_shards, 10):
        extra = os.path.join(DST_BASE, f"val_ablation_s{s}")
        if os.path.isdir(extra):
            shutil.rmtree(extra)
            print(f"  Removed extra shard dir: val_ablation_s{s}/")

    print(f"\nDone. {n_shards} shard dirs created under {DST_BASE}/")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    split(n)
