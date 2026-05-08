"""Figure C: Delay Ablation — per-sim-step batch ACI.

Three panels:
  (a) Threshold Convergence — Q_t over time (Proposition 2)
  (b) Empirical Coverage — rolling coverage over time (Proposition 2)
  (c) Tracking Error vs Delay — MAE |coverage - target| (Proposition 3)

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

DIR = os.path.dirname(os.path.abspath(__file__))

DELAYS = [0, 5, 10, 20, 100]
COLORS = {0: '#1f77b4', 5: '#ff7f0e', 10: '#2ca02c', 20: '#d62728', 100: '#9467bd'}
TAU_INIT = 0.48
TAU_OFFLINE = 0.257
WINDOW_LIGHT = 1
WINDOW_HEAVY = 100
ZOOM = 2000
TARGET = 0.9
BURN_IN = 300
TRACK_WINDOW = 50

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

mae_values = {}

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

    # Panel (a): threshold
    axes[0].plot(steps, tau_light, color=COLORS[d], linewidth=0.8, alpha=0.3)
    lw = 1.8 if d in (0, 100) else 1.4
    axes[0].plot(steps, tau_heavy, color=COLORS[d], linewidth=lw, alpha=0.9, label=f'K={d}')

    # Panel (b): coverage
    axes[1].plot(steps, cov_light, color=COLORS[d], linewidth=0.8, alpha=0.3)
    axes[1].plot(steps, cov_heavy, color=COLORS[d], linewidth=lw, alpha=0.9, label=f'K={d}')

    # Panel (c) data: tracking error after burn-in
    cov_roll = pd.Series(1.0 - err).rolling(TRACK_WINDOW, min_periods=TRACK_WINDOW).mean().values
    valid = cov_roll[BURN_IN:]
    valid = valid[~np.isnan(valid)]
    mae_values[d] = np.mean(np.abs(valid - TARGET))

# Panel (a) reference
axes[0].axhline(y=TAU_OFFLINE, color='black', linestyle='--', linewidth=1,
                label=r'offline $Q_{90\%}$')

# Panel (b) reference
axes[1].axhline(y=TARGET, color='black', linestyle='--', linewidth=1,
                label='target (90%)')

for ax in axes[:2]:
    ax.set_xscale('log')
    ax.set_xlabel('Simulation Step (log scale)', fontsize=13)
    ax.grid(True, alpha=0.2, which='both')

axes[0].set_ylabel(r'Threshold $Q_t$', fontsize=13)
axes[0].set_title(r'(a) Threshold Convergence', fontsize=14)
axes[0].legend(fontsize=10)

axes[1].set_ylabel('Coverage', fontsize=13)
axes[1].set_title('(b) Empirical Coverage', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].set_ylim(-0.1, 1.05)

# Panel (c): tracking error bar chart
ks = list(mae_values.keys())
maes = [mae_values[k] for k in ks]
bars = axes[2].bar([str(k) for k in ks], maes,
                   color=[COLORS[k] for k in ks], edgecolor='black', linewidth=0.6)
axes[2].set_xlabel(r'Expert Delay $\tau$', fontsize=13)
axes[2].set_ylabel(r'Mean $|$Coverage $- \, 0.9|$', fontsize=13)
axes[2].set_title('(c) Tracking Error vs. Delay', fontsize=14)
axes[2].grid(True, alpha=0.2, axis='y')
for bar, v in zip(bars, maes):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
for ext in ['png', 'pdf']:
    out = os.path.join(DIR, f'figure_c_batch.{ext}')
    plt.savefig(out, dpi=200)
    print(f'Saved: {out}')
plt.close()
