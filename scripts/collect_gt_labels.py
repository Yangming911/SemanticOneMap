"""
Collect GT semantic label statistics from HM3D scenes.
Loads the first N unique scenes and prints all category names with frequencies.
Usage:
    cd /opt/data/private/onemap/OneMap_obstacle
    conda run -n onemap python scripts/collect_gt_labels.py [--max-scenes 10]
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec

from eval.dataset_utils.hm3d_dataset import load_hm3d_episodes, load_hm3d_objects
from eval.dataset_utils import SceneData
from eval.semantic_collision import normalize_semantic_label, _NON_OBSTACLE_LABELS, _SEMANTIC_SAFETY_RADIUS_CELLS


OBJECT_NAV_PATH = "datasets/objectnav_hm3d_v1/val_mini/content/"
SCENE_PATH = "datasets/scene_datasets/"


def make_sim(scene_id: str) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = SCENE_PATH + scene_id
    backend_cfg.scene_dataset_config_file = (
        SCENE_PATH + "hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    )
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.hfov = 90
    rgb.position = np.array([0, 0.88, 0])
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [64, 64]  # small — we only need semantic info

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        action_space=dict(
            move_forward=ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        )
    )
    agent_cfg.sensor_specifications = [rgb]
    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return habitat_sim.Simulator(sim_cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-scenes", type=int, default=10, help="Number of unique scenes to scan")
    args = parser.parse_args()

    episodes, scene_data = load_hm3d_episodes([], {}, OBJECT_NAV_PATH)
    print(f"Total episodes: {len(episodes)}, unique scenes: {len(scene_data)}")

    all_labels: Counter = Counter()
    non_obstacle_hits: Counter = Counter()
    obs_dict_hits: Counter = Counter()
    other_hits: Counter = Counter()

    scenes_done = 0
    for scene_id in list(scene_data.keys())[: args.max_scenes]:
        print(f"\n[{scenes_done+1}/{args.max_scenes}] Loading scene: {scene_id}")
        try:
            sim = make_sim(scene_id)
        except Exception as e:
            print(f"  SKIP (sim init failed): {e}")
            continue

        scene_data = load_hm3d_objects(scene_data, sim.semantic_scene.objects, scene_id)
        sim.close()

        for raw_label, objs in scene_data[scene_id].object_locations.items():
            if not objs:
                continue
            norm = normalize_semantic_label(raw_label)
            all_labels[norm] += len(objs)
            if norm in _NON_OBSTACLE_LABELS:
                non_obstacle_hits[norm] += len(objs)
            elif norm in _SEMANTIC_SAFETY_RADIUS_CELLS:
                obs_dict_hits[norm] += len(objs)
            else:
                other_hits[norm] += len(objs)

        scenes_done += 1

    print("\n" + "=" * 60)
    print(f"GT Label Statistics ({scenes_done} scenes)")
    print("=" * 60)

    print(f"\n[Already in _SEMANTIC_SAFETY_RADIUS_CELLS] ({len(obs_dict_hits)} labels)")
    for label, cnt in obs_dict_hits.most_common():
        print(f"  {cnt:5d}  {label}")

    print(f"\n[Other labels — CANDIDATES for obstacle dict] ({len(other_hits)} labels)")
    for label, cnt in other_hits.most_common(50):
        print(f"  {cnt:5d}  {label}")

    print(f"\n[_NON_OBSTACLE_LABELS (skipped)] ({len(non_obstacle_hits)} labels)")
    for label, cnt in non_obstacle_hits.most_common(20):
        print(f"  {cnt:5d}  {label}")

    print("\n[Top-50 ALL labels]")
    for label, cnt in all_labels.most_common(50):
        tag = "(OBS)" if label in _SEMANTIC_SAFETY_RADIUS_CELLS else ("(NON)" if label in _NON_OBSTACLE_LABELS else "     ")
        print(f"  {cnt:5d}  {tag}  {label}")


if __name__ == "__main__":
    main()
