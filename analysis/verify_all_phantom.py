"""One-shot: classify all 10 pre-bug-fix collisions as PHANTOM vs SAME-FLOOR."""
import os, sys, json, subprocess
sys.path.insert(0, '/opt/data/private/OneMap4claude4')
from eval.dataset_utils.mp3d_dataset import load_mp3d_episodes

ROOT = '/opt/data/private/OneMap4claude4'
COLLS = [
    ('baseline', 9,  'chair',   400,  f'{ROOT}/results/mp3d_baseline_mini_dict3'),
    ('baseline', 11, 'table',   212,  f'{ROOT}/results/mp3d_baseline_mini_dict3'),
    ('baseline', 18, 'shower',  1442, f'{ROOT}/results/mp3d_baseline_mini_dict3'),
    ('baseline', 20, 'stool',   128,  f'{ROOT}/results/mp3d_baseline_mini_dict3'),
    ('baseline', 21, 'cabinet', 546,  f'{ROOT}/results/mp3d_baseline_mini_dict3'),
    ('wgate',    9,  'cabinet', 364,  f'{ROOT}/results/mp3d_wgate_mp3d_v5e_full_mini_pathA_add1_dict3'),
    ('wgate',    11, 'chair',   175,  f'{ROOT}/results/mp3d_wgate_mp3d_v5e_full_mini_pathA_add1_dict3'),
    ('wgate',    14, 'shower',  959,  f'{ROOT}/results/mp3d_wgate_mp3d_v5e_full_mini_pathA_add1_dict3'),
    ('wgate',    20, 'pillow',  1925, f'{ROOT}/results/mp3d_wgate_mp3d_v5e_full_mini_pathA_add1_dict3'),
    ('wgate',    21, 'sink',    319,  f'{ROOT}/results/mp3d_wgate_mp3d_v5e_full_mini_pathA_add1_dict3'),
]
eps, _ = load_mp3d_episodes([], {}, f'{ROOT}/datasets/objectnav_mp3d_v1/val_mini/content/')
by_scene = {}
for run, ep, cause, step, rdir in COLLS:
    sc = eps[ep].scene_id
    by_scene.setdefault(sc, []).append({'run':run,'ep':ep,'cause':cause,'step':step,'rdir':rdir})

WORKER = f'{ROOT}/analysis/_verify_worker.py'

PY = '/opt/conda/envs/onemap/bin/python'
all_r = []
for sc, items in by_scene.items():
    out = f'/tmp/vp_{abs(hash(sc))}.json'
    if os.path.exists(out): os.remove(out)
    r = subprocess.run(['xvfb-run','-a',PY,WORKER,sc,json.dumps(items),out],
                       capture_output=True, text=True, timeout=120)
    if not os.path.exists(out):
        print(f'FAIL {sc}: {r.stderr[-200:]}', file=sys.stderr); continue
    all_r.extend(json.load(open(out)))

print('\n' + '='*92)
print(f'{"run":<10} {"ep":>3} {"cause":>9} {"step":>5} {"d_to_judge_obj":>15} {"d_below":>9}  verdict')
print('-'*92)
counts = {'baseline':{'SAME-FLOOR':0,'PHANTOM':0},'wgate':{'SAME-FLOOR':0,'PHANTOM':0}}
for r in sorted(all_r, key=lambda r:(r['run'],r['ep'])):
    print(f'{r["run"]:<10} {r["ep"]:>3} {r["cause"]:>9} {r["step"]:>5} {r["d"]:>15.3f} {r["d_below"]:>9.2f}  {r["verdict"]}')
    if r['verdict'] in counts[r['run']]:
        counts[r['run']][r['verdict']] += 1
print('\nSummary:')
for run in counts:
    print(f'  {run}: same-floor={counts[run]["SAME-FLOOR"]}, phantom={counts[run]["PHANTOM"]}')
