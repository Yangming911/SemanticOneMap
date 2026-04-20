"""Phase 1 survey — per scene.

For each baseline trajectory in this scene, walk poses and record:
  - per_obj_min: closest agent approach distance per (same-floor) GT object
  - per_step_close: per step, list of (class, obj_id, d_m) for nearby objects

Same-floor filter (★ INCLUDES the new BUG-FIX lower-floor bound):
  - bot_y - floor_y > 0.3   → skip (object on shelf / floating)
  - ctr_y - floor_y > 1.5   → skip (mounted on wall / ceiling)
  - floor_y - bot_y > 0.5   → skip ★ NEW: object on lower floor (phantom bug)

Args:
  sys.argv[1] = scene_id (e.g. 'mp3d/X7HyMhZNoso/X7HyMhZNoso.glb')
  sys.argv[2] = JSON list of {"ep_id": int, "floor_y": float, "poses_path": str}
  sys.argv[3] = out_path (JSON)
"""
import os, sys, json
import numpy as np

sys.path.insert(0, '/opt/data/private/OneMap4claude4')
import habitat_sim
from eval.dataset_utils.mp3d_dataset import load_mp3d_episodes, load_mp3d_objects
from eval.semantic_collision import normalize_semantic_label, _NON_OBSTACLE_LABELS

scene_id = sys.argv[1]
ep_items = json.loads(sys.argv[2])
out_path = sys.argv[3]

# Need scene_data structure populated first via mp3d loader
episodes, scene_data = load_mp3d_episodes(
    [], {},
    '/opt/data/private/OneMap4claude4/datasets/objectnav_mp3d_v1/val_mini/content/',
)

# Init habitat-sim with minimal RGB sensor (avoids teardown segfault)
full = f'/opt/data/private/OneMap4claude4/datasets/scene_datasets/{scene_id}'
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = full
rgb = habitat_sim.CameraSensorSpec()
rgb.uuid = 'rgb'
rgb.sensor_type = habitat_sim.SensorType.COLOR
rgb.resolution = [128, 128]
rgb.position = np.array([0, 0.88, 0])
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [rgb]
sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))
load_mp3d_objects(scene_data, sim.semantic_scene.objects, scene_id)

# ── Constants ───────────────────────────────────────────────────
CELL = 0.1
DIST_CAP_M = 1.5         # only record events within this distance (m)
ABOVE_BOT = 0.3          # bot_y - floor_y > this  → skip (shelf / floating)
ABOVE_CTR = 1.5          # ctr_y - floor_y > this  → skip (wall mount / ceiling)
BELOW_BOT = 0.5          # floor_y - bot_y > this  → skip ★ NEW: phantom-cross-floor bug fix
# ─────────────────────────────────────────────────────────────────

# Pre-flatten ALL same-class candidate objects (apply only non-floor filters here)
all_objs = []  # list of dicts, with bot_y/ctr_y for per-ep floor filter
for raw_cat, objs in scene_data[scene_id].object_locations.items():
    norm_cat = normalize_semantic_label(raw_cat)
    if norm_cat in _NON_OBSTACLE_LABELS:
        continue
    for obj in objs:
        c = np.asarray(obj.bbox.center, dtype=np.float32)
        s = np.asarray(obj.bbox.sizes, dtype=np.float32)
        if c.shape[0] < 3:
            continue
        cx_ = -float(c[2])
        cy_ = -float(c[0])
        ctr_y = float(c[1])
        bot_y = ctr_y - float(s[1]) / 2.0
        all_objs.append({
            'raw_cat': raw_cat,
            'norm_cat': norm_cat,
            'obj_id': obj.object_id,
            'cx': cx_, 'cy': cy_,
            'ctr_y': ctr_y, 'bot_y': bot_y,
        })

ep_results = {}
for item in ep_items:
    ep_id = int(item['ep_id'])
    floor_y = float(item['floor_y'])
    poses_path = item['poses_path']

    poses = np.genfromtxt(poses_path, delimiter=',')
    if poses.ndim == 1:
        poses = poses.reshape(1, -1)
    n_steps = len(poses)

    # Apply ALL THREE same-floor filters per episode (floor_y is per-ep)
    same_floor = []
    for o in all_objs:
        if (o['bot_y'] - floor_y) > ABOVE_BOT:
            continue
        if (o['ctr_y'] - floor_y) > ABOVE_CTR:
            continue
        if (floor_y - o['bot_y']) > BELOW_BOT:    # ★ NEW BUG FIX
            continue
        same_floor.append(o)

    # Convert to numpy for vectorised distance per step
    if same_floor:
        obj_xy = np.array([[o['cx'], o['cy']] for o in same_floor], dtype=np.float32)  # (N, 2)
    else:
        obj_xy = np.zeros((0, 2), dtype=np.float32)

    per_obj_min = {}      # obj_id → {norm_cat, cx, cy, bot_y, ctr_y, d_min, t_min}
    per_step_close = []   # list of {step, agent_xy, near: [{norm_cat, obj_id, d_m}, ...]}

    for t in range(n_steps):
        ax, ay = float(poses[t, 0]), float(poses[t, 1])
        if obj_xy.shape[0] == 0:
            continue
        diffs = obj_xy - np.array([ax, ay], dtype=np.float32)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])

        # update per_obj_min
        for i, d in enumerate(dists):
            key = same_floor[i]['obj_id']
            d_f = float(d)
            if key not in per_obj_min or d_f < per_obj_min[key]['d_min']:
                o = same_floor[i]
                per_obj_min[key] = {
                    'norm_cat': o['norm_cat'],
                    'raw_cat': o['raw_cat'],
                    'cx': o['cx'], 'cy': o['cy'],
                    'bot_y': o['bot_y'], 'ctr_y': o['ctr_y'],
                    'd_min': d_f, 't_min': t,
                }

        # collect near objects for this step (within CAP)
        near_idx = np.where(dists <= DIST_CAP_M)[0]
        if near_idx.size > 0:
            # per-class: keep only the CLOSEST obj per norm_cat to keep file small
            by_class = {}
            for i in near_idx:
                cat = same_floor[i]['norm_cat']
                d_f = float(dists[i])
                if cat not in by_class or d_f < by_class[cat][1]:
                    by_class[cat] = (same_floor[i]['obj_id'], d_f)
            per_step_close.append({
                'step': t,
                'agent_xy': [ax, ay],
                'near': [{'norm_cat': c, 'obj_id': oid, 'd_m': d}
                         for c, (oid, d) in by_class.items()],
            })

    # Filter per_obj_min to only objects with d_min < CAP (don't bloat output)
    per_obj_min_kept = {oid: v for oid, v in per_obj_min.items() if v['d_min'] <= DIST_CAP_M}

    ep_results[str(ep_id)] = {
        'scene_id': scene_id,
        'floor_y': floor_y,
        'n_steps': n_steps,
        'n_same_floor_objs': len(same_floor),
        'per_obj_min': per_obj_min_kept,
        'per_step_close': per_step_close,
    }

with open(out_path, 'w') as f:
    json.dump(ep_results, f)
sim.close()
print(f'OK scene={scene_id} eps={len(ep_results)}', file=sys.stderr)
