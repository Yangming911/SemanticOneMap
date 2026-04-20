"""Per-scene phantom verification worker."""
import os, sys, json
import numpy as np
sys.path.insert(0, '/opt/data/private/OneMap4claude4')
import habitat_sim
from eval.dataset_utils.mp3d_dataset import load_mp3d_episodes, load_mp3d_objects
from eval.semantic_collision import normalize_semantic_label

scene_id = sys.argv[1]
items = json.loads(sys.argv[2])
out_path = sys.argv[3]

eps, scene_data = load_mp3d_episodes([], {},
    '/opt/data/private/OneMap4claude4/datasets/objectnav_mp3d_v1/val_mini/content/')

bc = habitat_sim.SimulatorConfiguration()
bc.scene_id = f'/opt/data/private/OneMap4claude4/datasets/scene_datasets/{scene_id}'
rgb = habitat_sim.CameraSensorSpec()
rgb.uuid = 'rgb'
rgb.sensor_type = habitat_sim.SensorType.COLOR
rgb.resolution = [128, 128]
rgb.position = np.array([0, 0.88, 0])
ac = habitat_sim.agent.AgentConfiguration()
ac.sensor_specifications = [rgb]
sim = habitat_sim.Simulator(habitat_sim.Configuration(bc, [ac]))
load_mp3d_objects(scene_data, sim.semantic_scene.objects, scene_id)

res = []
for it in items:
    floor_y = float(eps[it['ep']].start_position[1])
    poses = np.genfromtxt(f"{it['rdir']}/trajectories/poses_{it['ep']}.csv", delimiter=',')
    step = min(it['step'], len(poses) - 1)
    ax, ay = float(poses[step][0]), float(poses[step][1])
    cn = normalize_semantic_label(it['cause'])
    cands = []
    for cat, objs in scene_data[scene_id].object_locations.items():
        if normalize_semantic_label(cat) != cn:
            continue
        for o in objs:
            c = np.asarray(o.bbox.center, dtype=np.float32)
            s = np.asarray(o.bbox.sizes, dtype=np.float32)
            if c.shape[0] < 3:
                continue
            cx_ = -float(c[2])
            cy_ = -float(c[0])
            ctr_y = float(c[1])
            bot_y = ctr_y - float(s[1]) / 2.0
            if (bot_y - floor_y) > 0.3:
                continue
            if (ctr_y - floor_y) > 1.5:
                continue
            d = float(np.hypot(cx_ - ax, cy_ - ay))
            cands.append({'d': d, 'bot_y': bot_y})
    if not cands:
        res.append({**it, 'd': -1.0, 'd_below': 0.0, 'verdict': 'NO_CAND'})
        continue
    n = min(cands, key=lambda c: c['d'])
    db = floor_y - n['bot_y']
    if db > 0.5:
        v = 'PHANTOM'
    elif db < -0.3:
        v = 'OVERHEAD'
    else:
        v = 'SAME-FLOOR'
    res.append({**it, 'd': n['d'], 'd_below': db, 'verdict': v})

with open(out_path, 'w') as f:
    json.dump(res, f)
sim.close()
