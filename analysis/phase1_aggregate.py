"""Phase 1c — aggregate the per-ep survey into dict-design statistics.

Two outputs:
  (A) Per-class d_min histogram across all (ep, obj_id) close approaches
  (B) Given a candidate dict, simulate first-hit per ep, report would-collide ep counts

The simulation is approximate (uses Euclidean d ≤ radius*CELL); the real
evaluator rasterises seed pixel + cv2.MORPH_ELLIPSE dilation, so add ±0.5
cell tolerance. For Phase 1 design this is plenty accurate.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = '/opt/data/private/OneMap4claude4'
PASSES = f'{ROOT}/analysis/phase1_baseline_passes.json'

with open(PASSES) as f:
    data = json.load(f)

CELL = 0.1

# ── (A) per-class d_min distribution across all (ep, obj_id) ─────────────
per_class_dmin = defaultdict(list)   # norm_cat → [d_min_m, ...]
per_class_counts = defaultdict(lambda: defaultdict(set))  # norm_cat → "<r m" → {ep_id}
for ep_id, info in data.items():
    for oid, rec in info['per_obj_min'].items():
        per_class_dmin[rec['norm_cat']].append(rec['d_min'])
        for r_cells in (1, 2, 3, 4, 5, 6):
            r_m = r_cells * CELL + 0.05    # +0.5 cell tolerance for raster check
            if rec['d_min'] <= r_m:
                per_class_counts[rec['norm_cat']][r_cells].add(int(ep_id))

print('=' * 90)
print('(A) Per-class closest-approach distribution across 33 baseline trajectories')
print('=' * 90)
print(f'{"class":<22s} {"n_obj":>5s} {"min":>6s} {"5%":>6s} {"25%":>6s} {"med":>6s} '
      f'{"75%":>6s}  | n_ep that come within radius cells [1, 2, 3, 4, 5]')
print('-' * 90)
classes_sorted = sorted(per_class_dmin.keys(),
                        key=lambda c: -len(per_class_counts[c].get(3, set())))
for c in classes_sorted:
    arr = np.array(per_class_dmin[c])
    pcts = np.percentile(arr, [0, 5, 25, 50, 75])
    cnts = ' '.join(f'{len(per_class_counts[c].get(r, set())):>2d}' for r in (1, 2, 3, 4, 5))
    print(f'{c:<22s} {len(arr):>5d} {pcts[0]:>6.2f} {pcts[1]:>6.2f} '
          f'{pcts[2]:>6.2f} {pcts[3]:>6.2f} {pcts[4]:>6.2f}  | {cnts}')

# ── (B) Simulate first-hit per ep under various candidate dicts ──────────
def simulate_first_hit(dict_radii, allowed_above_classes=None):
    """For each ep, find the FIRST step where any object in dict_radii is hit.
    Returns: {ep_id: (step, class, obj_id, d_m) or None if no collision}.
    """
    results = {}
    for ep_id, info in data.items():
        first = None
        for entry in info['per_step_close']:
            for near in entry['near']:
                cls = near['norm_cat']
                if cls not in dict_radii:
                    continue
                r_m = dict_radii[cls] * CELL + 0.05
                if near['d_m'] <= r_m:
                    if first is None or entry['step'] < first['step']:
                        first = {'step': entry['step'], 'class': cls,
                                 'obj_id': near['obj_id'], 'd_m': near['d_m']}
                    break  # one per step is enough
            if first is not None:
                break
        results[ep_id] = first
    return results

def report(name, dict_radii):
    res = simulate_first_hit(dict_radii)
    n_coll = sum(1 for v in res.values() if v is not None)
    by_cls = defaultdict(int)
    for v in res.values():
        if v: by_cls[v['class']] += 1
    print(f'\n  → "{name}": {n_coll}/33 collisions, by class: {dict(sorted(by_cls.items(), key=lambda kv: -kv[1]))}')
    eps_hit = sorted(int(k) for k, v in res.items() if v is not None)
    print(f'     ep_ids: {eps_hit}')
    return res

# Candidate dicts to evaluate
print('\n' + '=' * 90)
print('(B) First-hit simulation under candidate dicts')
print('=' * 90)

# current dict3 (for reference; phantom-filtered → very few collisions)
dict3_current = {
    'toilet': 3, 'pillow': 2, 'lamp': 2, 'bed': 3, 'sofa': 3, 'couch': 3,
    'chest of drawers': 2, 'cabinet': 1, 'sink': 2, 'bathtub': 3, 'counter': 2,
    'fireplace': 3, 'stool': 2, 'tv': 2, 'shower': 2, 'cushion': 1,
    'table': 3, 'chair': 1,
}
report('dict3_current (with bug-fix)', dict3_current)

# Tight: minimum that catches a few common close-approach classes
dict_tight = {k: 2 for k in dict3_current}
report('all=2 cells (0.20m)', dict_tight)

dict_med = {k: 3 for k in dict3_current}
report('all=3 cells (0.30m)', dict_med)

dict_wide = {k: 4 for k in dict3_current}
report('all=4 cells (0.40m)', dict_wide)

dict_v_wide = {k: 5 for k in dict3_current}
report('all=5 cells (0.50m)', dict_v_wide)

# user's proposed dict4
dict4 = {
    'shower':           3,
    'cabinet':          4,
    'chest of drawers': 4,
    'table':            5,
    'tv':               2,
}
report('dict4 (proposed)', dict4)

# Target shape: per-class radius from d_min distribution at 25th percentile
# This means: top quarter of objects in the class would be 'collision'
print('\n' + '=' * 90)
print('(C) Per-class radius proposal: ceil(p25(d_min)/CELL) clipped to [2, 5]')
print('=' * 90)
proposed = {}
for c in dict3_current:
    if c not in per_class_dmin:
        proposed[c] = 2
        continue
    p25 = np.percentile(per_class_dmin[c], 25)
    proposed[c] = max(2, min(5, int(np.ceil(p25 / CELL))))
print(f'  proposed: {proposed}')
report('proposed_p25', proposed)

# Save full results
out_summary = {
    'per_class_dmin_stats': {c: {
        'n_obj': len(per_class_dmin[c]),
        'min': float(np.min(per_class_dmin[c])),
        'p5': float(np.percentile(per_class_dmin[c], 5)),
        'p25': float(np.percentile(per_class_dmin[c], 25)),
        'p50': float(np.percentile(per_class_dmin[c], 50)),
        'p75': float(np.percentile(per_class_dmin[c], 75)),
        'eps_within_r=1': sorted(per_class_counts[c].get(1, set())),
        'eps_within_r=2': sorted(per_class_counts[c].get(2, set())),
        'eps_within_r=3': sorted(per_class_counts[c].get(3, set())),
        'eps_within_r=4': sorted(per_class_counts[c].get(4, set())),
        'eps_within_r=5': sorted(per_class_counts[c].get(5, set())),
    } for c in classes_sorted},
}
with open(f'{ROOT}/analysis/phase1_class_stats.json', 'w') as f:
    json.dump(out_summary, f, indent=1)
print(f'\nSaved class stats to {ROOT}/analysis/phase1_class_stats.json')
