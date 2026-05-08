"""Figure A: SCR vs Coverage Target — Pareto frontier.

Compares OACP (ACI online), Fixed-CP (offline threshold), Argmax, and Baseline.
OACP and Fixed-CP shown as PCHIP-interpolated curves through cov=20/40/60/80.
Argmax and Baseline shown as reference markers with horizontal dashed lines.
Broken y-axis compresses the gap between Baseline (~31%) and the CP methods (~8-12%).

Data: figure_a_data.csv (all 330-episode val_ablation results, open-vocab + 5 holdout).

Usage: python plot_figure_a.py
Output: figure_a.png / .pdf
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(DIR, 'figure_a_data.csv'))

# Extract data
baseline = df[df['method'] == 'Baseline'].iloc[0]
argmax = df[df['method'] == 'Argmax'].iloc[0]

oacp_048 = df[(df['method'] == 'OACP') & (df['tau_init'] == 0.48)].sort_values('coverage')
fixed = df[df['method'] == 'Fixed-CP'].sort_values('coverage')

# Filter to cov = 20, 40, 60, 80
oacp_pts = oacp_048[oacp_048['coverage'].isin([20, 40, 60, 80])]
fixed_pts = fixed[fixed['coverage'].isin([20, 40, 60, 80])]

# PCHIP interpolation
x_smooth = np.linspace(20, 80, 200)
oacp_interp = PchipInterpolator(oacp_pts['coverage'].values, oacp_pts['SCR'].values)
fixed_interp = PchipInterpolator(fixed_pts['coverage'].values, fixed_pts['SCR'].values)

# --- Plot ---
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={'height_ratios': [0.6, 3], 'hspace': 0.06})

# Bottom panel: OACP, Fixed-CP, Argmax
ax_bot.plot(x_smooth, fixed_interp(x_smooth), '-', color='#2196F3', linewidth=3, zorder=4, label='Offline-CP')
ax_bot.plot(x_smooth, oacp_interp(x_smooth), '-', color='#9C27B0', linewidth=3, zorder=4, label='OACP')
ax_bot.axhline(y=argmax['SCR'], color='#4CAF50', linestyle='--', linewidth=1.8, alpha=0.7, zorder=2)
ax_bot.scatter([50], [argmax['SCR']], marker='s', s=130, c='#4CAF50', zorder=6, label='Argmax')

ax_bot.set_ylim(6, 14.5)
ax_bot.set_ylabel('Semantic Collision Rate', fontsize=22)
ax_bot.set_xlabel('Coverage Target', fontsize=22)
ax_bot.set_xticks([])
ax_bot.set_yticks([argmax['SCR'], 7.47])
ax_bot.set_yticklabels(['12.4%', '7.5%'], fontsize=18, color='#555555')
ax_bot.tick_params(axis='y', length=3, color='#999999')
ax_bot.grid(False)

# Top panel: Baseline
ax_top.axhline(y=baseline['SCR'], color='#FF9800', linestyle='--', linewidth=1.8, alpha=0.7, zorder=2)
ax_top.scatter([50], [baseline['SCR']], marker='*', s=250, c='#FF9800', zorder=6, label='Baseline')
ax_top.set_ylim(29, 33)
ax_top.set_xticks([])
ax_top.set_yticks([baseline['SCR']])
ax_top.set_yticklabels(['31.2%'], fontsize=18, color='#555555')
ax_top.tick_params(axis='y', length=3, color='#999999')
ax_top.grid(False)

# Combined legend: Baseline, Argmax, Offline-CP, OACP
handles_top, labels_top = ax_top.get_legend_handles_labels()
handles_bot, labels_bot = ax_bot.get_legend_handles_labels()
all_h = handles_top + handles_bot
all_l = labels_top + labels_bot
order = ['Baseline', 'Argmax', 'Offline-CP', 'OACP']
ordered_h = [all_h[all_l.index(l)] for l in order]
ordered_l = order
ax_bot.legend(ordered_h, ordered_l,
              loc='upper right', fontsize=16, frameon=False,
              bbox_to_anchor=(0.98, 1.14))

# Clean spines: only left and bottom axes
for ax in [ax_top, ax_bot]:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['bottom'].set_color('#333333')

ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(bottom=False)

# Broken axis marks (left side only)
d = 0.015
kwargs = dict(transform=ax_top.transAxes, color='#333333', clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1-d, 1+d), **kwargs)
ax_bot.set_xlim(15, 85)

# Axis arrows
ax_bot.annotate('', xy=(87, ax_bot.get_ylim()[0]), xytext=(15, ax_bot.get_ylim()[0]),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0, mutation_scale=20),
                annotation_clip=False)
ax_top.annotate('', xy=(ax_top.get_xlim()[0], 34), xytext=(ax_top.get_xlim()[0], 29),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2.0, mutation_scale=20),
                annotation_clip=False)

for ext in ['png', 'pdf']:
    out = os.path.join(DIR, f'figure_a.{ext}')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close()
