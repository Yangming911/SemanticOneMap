"""Phase 1 orchestrator: batch baseline trajectories by scene, run survey per scene
via xvfb-run subprocess (avoids habitat-sim teardown segfaults).

Output: analysis/phase1_baseline_passes.json — combined per-ep results.
"""
import os, sys, json, subprocess
sys.path.insert(0, '/opt/data/private/OneMap4claude4')
from eval.dataset_utils.mp3d_dataset import load_mp3d_episodes

ROOT = '/opt/data/private/OneMap4claude4'
RESULTS_DIR = f'{ROOT}/results/mp3d_baseline_mini_dict3'
OUT = f'{ROOT}/analysis/phase1_baseline_passes.json'

# Load 33 mp3d_mini episodes (same casefold sort as evaluator)
episodes, _ = load_mp3d_episodes([], {},
    f'{ROOT}/datasets/objectnav_mp3d_v1/val_mini/content/')

# Build per-ep records, group by scene
by_scene = {}
for ep in episodes:
    poses_path = f'{RESULTS_DIR}/trajectories/poses_{ep.episode_id}.csv'
    if not os.path.exists(poses_path):
        print(f'  ep {ep.episode_id}: missing trajectory, skipping', file=sys.stderr)
        continue
    item = {
        'ep_id': ep.episode_id,
        'floor_y': float(ep.start_position[1]),
        'poses_path': poses_path,
    }
    by_scene.setdefault(ep.scene_id, []).append(item)

print(f'Scenes: {len(by_scene)} (total eps: {sum(len(v) for v in by_scene.values())})',
      file=sys.stderr)

PYBIN = '/opt/conda/envs/onemap/bin/python'
SCRIPT = f'{ROOT}/analysis/phase1_survey_one_scene.py'

combined = {}
for scene_id, items in by_scene.items():
    print(f'\n=== {scene_id} ({len(items)} ep) ===', file=sys.stderr)
    out_path = f'/tmp/phase1_scene_{abs(hash(scene_id))}.json'
    if os.path.exists(out_path): os.remove(out_path)
    res = subprocess.run(
        ['xvfb-run', '-a', PYBIN, SCRIPT, scene_id, json.dumps(items), out_path],
        capture_output=True, text=True, timeout=600,
    )
    if not os.path.exists(out_path):
        print(f'  FAIL rc={res.returncode}; stderr tail:\n{res.stderr[-600:]}',
              file=sys.stderr)
        continue
    with open(out_path) as f:
        scene_results = json.load(f)
    print(f'  ok: {len(scene_results)} ep', file=sys.stderr)
    combined.update(scene_results)

with open(OUT, 'w') as f:
    json.dump(combined, f)
print(f'\nWrote {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB, {len(combined)} eps)',
      file=sys.stderr)
