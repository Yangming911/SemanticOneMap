"""Figure C: Delay Ablation — per-sim-step batch ACI.

Two panels; panel (b) has marginal box plots on right edge:
  (a) Threshold Convergence — Q_t over time (Proposition 2)
  (b) Empirical Coverage + right-margin box plots (Proposition 3)

Data: ep24 from shard 3 (val_ablation), per-sim-step batch calibration.
K=0 uses delay=1 as proxy (batch logging, negligible difference).

Usage: python plot_figure_c_batch.py
Output: figure_c_batch.png / .pdf
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DIR = os.path.dirname(os.path.abspath(__file__))

DELAYS = [0, 5, 10, 20, 100]
COLORS = {0: '#d62728', 5: '#ff7f0e', 10: '#2ca02c', 20: '#1f77b4', 100: '#9467bd'}
TAU_INIT = 0.48
TAU_OFFLINE = 0.257
WINDOW_LIGHT = 1
WINDOW_HEAVY = 100
ZOOM = 2000
TARGET = 0.9
BURN_IN = 300
TRACK_WINDOW = 50

fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[5, 4.2, 0.8], wspace=0.35)
gs_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1:],
                                            width_ratios=[5, 1], wspace=0.02)

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs_right[0])
ax_box = fig.add_subplot(gs_right[1], sharey=ax_b)

cov_distributions = {}

for d in DELAYS:
    csv_path = os.path.join(DIR, f'ep24_batch_delay{d}.csv')
    df = pd.read_csv(csv_path)
    if 'episode_id' in df.columns:
        df = df[df['episode_id'] == 24]
    df = df.head(ZOOM).reset_index(drop=True)

    tau = df['tau_after'].values
    err = df['err'].values

    k_prepend = max(1, d)
    tau = np.concatenate([np.full(k_prepend, TAU_INIT), tau])[:ZOOM]
    err = np.concatenate([np.full(k_prepend, 1.0), err])[:ZOOM]
    steps = np.arange(1, len(tau) + 1)

    tau_light = pd.Series(tau).rolling(WINDOW_LIGHT, min_periods=1).mean().values
    cov_light = pd.Series(1.0 - err).rolling(WINDOW_LIGHT, min_periods=1).mean().values
    tau_heavy = pd.Series(tau).rolling(WINDOW_HEAVY, min_periods=1).mean().values
    cov_heavy = pd.Series(1.0 - err).rolling(WINDOW_HEAVY, min_periods=1).mean().values

    lw = 1.8 if d in (0, 100) else 1.4

    # Panel (a): threshold
    ax_a.plot(steps, tau_light, color=COLORS[d], linewidth=0.8, alpha=0.3)
    ax_a.plot(steps, tau_heavy, color=COLORS[d], linewidth=lw, alpha=0.9, label=r'$\tau$={}'.format(d))

    # Panel (b): coverage
    ax_b.plot(steps, cov_light, color=COLORS[d], linewidth=0.8, alpha=0.3)
    ax_b.plot(steps, cov_heavy, color=COLORS[d], linewidth=lw, alpha=0.9, label=r'$\tau$={}'.format(d))

    # Coverage distribution after convergence (for box plots)
    cov_roll = pd.Series(1.0 - err).rolling(TRACK_WINDOW, min_periods=TRACK_WINDOW).mean().values
    valid = cov_roll[BURN_IN:]
    valid = valid[~np.isnan(valid)]
    cov_distributions[d] = valid

# Panel (a) styling
ax_a.axhline(y=TAU_OFFLINE, color='black', linestyle='--', linewidth=1,
             label=r'offline $Q_{90\%}$')
ax_a.set_xscale('log')
ax_a.set_xlabel('Simulation Step (log scale)', fontsize=13)
ax_a.set_ylabel(r'Threshold $Q_t$', fontsize=13)
ax_a.set_title(r'(a) Threshold Convergence', fontsize=14)
ax_a.legend(fontsize=10, frameon=False)
ax_a.grid(False)

# Panel (b) styling
ax_b.axhline(y=TARGET, color='black', linestyle='--', linewidth=1,
             label='target (90%)')
ax_b.set_xscale('log')
ax_b.set_xlabel('Simulation Step (log scale)', fontsize=13)
ax_b.set_ylabel('Coverage', fontsize=13)
ax_b.set_title('(b) Empirical Coverage', fontsize=14)
ax_b.legend(fontsize=10, loc='lower right', frameon=False)
ax_b.set_ylim(-0.05, 1.05)
ax_b.grid(False)

# --- Right margin: vertical box plots showing coverage distribution ---
box_data = [cov_distributions[d] for d in DELAYS]
positions = list(range(len(DELAYS)))

bp = ax_box.boxplot(box_data, vert=True, positions=positions, widths=0.65,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8, color='#555555'),
                    capprops=dict(linewidth=0.8, color='#555555'),
                    boxprops=dict(linewidth=0.5))
for patch, d in zip(bp['boxes'], DELAYS):
    patch.set_facecolor(COLORS[d])
    patch.set_alpha(0.7)

ax_box.axhline(y=TARGET, color='black', linestyle='--', linewidth=1)
ax_box.set_xticks(positions)
ax_box.set_xticklabels([str(d) for d in DELAYS], fontsize=8)
ax_box.set_xlabel(r'$\tau$', fontsize=10)
ax_box.tick_params(axis='y', labelleft=False)
ax_box.grid(False)

for ext in ['png', 'pdf']:
    out = os.path.join(DIR, f'figure_c_batch.{ext}')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')
plt.close()
