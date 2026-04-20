"""
Analyze OACP calibration results and generate figures for the paper.

Figures produced:
  figures/tau_convergence.png      — τ_t trajectory over calibration steps (Theorem 1)
  figures/coverage_convergence.png — running average of err_t → α_target (Theorem 1)

Usage:
  conda run -n onemap python analysis/plot_oacp_results.py [--oacp-csv results/gclip_oacp/oacp_calibration_*.csv]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA_TARGET = 0.1   # 1 - target_coverage


# ── helpers ──────────────────────────────────────────────────────────────────

def load_nav_log(log_path: str):
    """Parse eval log file for SR / SPL / collision counts."""
    sr, spl, collisions, n_eps = None, None, None, None
    if not os.path.exists(log_path):
        return sr, spl, collisions, n_eps
    with open(log_path) as f:
        for line in f:
            if "Overall success" in line:
                try:
                    sr = float(line.split("Overall success:")[1].split(",")[0].strip())
                except Exception:
                    pass
            if "SPL" in line and "=" in line:
                try:
                    spl = float(line.split("SPL =")[1].split()[0].strip().rstrip(","))
                except Exception:
                    pass
            if "semantic collisions:" in line.lower():
                try:
                    collisions = int(line.lower().split("semantic collisions:")[1].split(",")[0].strip())
                except Exception:
                    pass
    return sr, spl, collisions, n_eps


def load_oacp_csv(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  No files matching: {pattern}")
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"  Loaded {len(df)} calibration events from {len(files)} file(s)")
    return df


# ── Figure 1: τ_t convergence ────────────────────────────────────────────────

def plot_tau_convergence(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Per-episode curves (light)
    for ep_id, grp in df.groupby("episode_id"):
        ax.plot(grp["step"].values, grp["tau"].values,
                color="steelblue", alpha=0.25, linewidth=0.8)

    # Global mean across episodes (bold)
    mean_tau = df.groupby("step")["tau"].mean()
    ax.plot(mean_tau.index, mean_tau.values,
            color="steelblue", linewidth=2.5, label=r"Mean $\tau_t$")

    ax.set_xlabel("Calibration step $t$", fontsize=12)
    ax.set_ylabel(r"Threshold $\tau_t$ (cosine similarity)", fontsize=12)
    ax.set_title(r"ACI Threshold Convergence (Theorem 1)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: running coverage rate ──────────────────────────────────────────

def plot_coverage_convergence(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # Per-episode running mean of (1 - err_t)
    for ep_id, grp in df.groupby("episode_id"):
        cov = 1 - grp["err"].expanding().mean().values
        ax.plot(grp["step"].values, cov,
                color="darkorange", alpha=0.25, linewidth=0.8)

    # Global running mean
    global_err = []
    for _, grp in df.groupby("episode_id"):
        global_err.extend(grp["err"].tolist())
    global_cov = 1 - np.array(global_err)
    running_mean = np.cumsum(global_cov) / (np.arange(len(global_cov)) + 1)
    ax.plot(np.arange(len(running_mean)), running_mean,
            color="darkorange", linewidth=2.5, label="Running coverage rate")

    ax.axhline(1 - ALPHA_TARGET, color="red", linestyle="--", linewidth=1.5,
               label=rf"Target $1-\alpha={1-ALPHA_TARGET:.0%}$")

    ax.set_xlabel("Cumulative calibration step", fontsize=12)
    ax.set_ylabel("Coverage rate", fontsize=12)
    ax.set_title("ACI Coverage Convergence (Theorem 1)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(df_oacp: pd.DataFrame):
    print("\n" + "="*72)
    print("OACP Calibration Summary")
    print("="*72)

    if df_oacp.empty:
        print("  No OACP data available.")
        return

    n_episodes = df_oacp["episode_id"].nunique()
    n_calls = len(df_oacp)
    coverage = 1 - df_oacp["err"].mean()
    final_tau = df_oacp.groupby("episode_id")["tau"].last().mean()
    initial_tau = df_oacp.groupby("episode_id")["tau"].first().mean()

    print(f"  Episodes:         {n_episodes}")
    print(f"  Total calib calls:{n_calls}")
    print(f"  Coverage rate:    {coverage:.3f}  (target={1-ALPHA_TARGET:.1f})")
    print(f"  Initial τ (mean): {initial_tau:.4f}")
    print(f"  Final τ (mean):   {final_tau:.4f}")

    # Per-episode
    per_ep = df_oacp.groupby("episode_id").agg(
        n_calls=("step", "count"),
        coverage=("err", lambda x: 1 - x.mean()),
        final_tau=("tau", "last"),
    )
    print(f"\n  Per-episode stats (first 10):")
    print(per_ep.head(10).to_string())
    print("="*72)


# ── Navigation comparison table ───────────────────────────────────────────────

def print_nav_table(results: list):
    """results: list of (name, sr, collision_rate, coverage)"""
    print("\n" + "="*72)
    print("Navigation Results Comparison")
    print(f"{'Method':<30} {'SR':>6} {'Collision':>10} {'Coverage':>10}")
    print("-"*72)
    for name, sr, coll, cov in results:
        sr_s = f"{sr:.1%}" if sr is not None else "N/A"
        coll_s = f"{coll:.1%}" if coll is not None else "N/A"
        cov_s = f"{cov:.1%}" if cov is not None else "N/A"
        print(f"{name:<30} {sr_s:>6} {coll_s:>10} {cov_s:>10}")
    print("="*72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oacp-csv", default="results/gclip_oacp/oacp_calibration_*.csv")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading OACP calibration data...")
    df_oacp = load_oacp_csv(args.oacp_csv)

    if not df_oacp.empty:
        plot_tau_convergence(df_oacp, os.path.join(args.out_dir, "tau_convergence.png"))
        plot_coverage_convergence(df_oacp, os.path.join(args.out_dir, "coverage_convergence.png"))
        print_summary_table(df_oacp)

    # Navigation comparison (hard-coded from known results + logs)
    # Baseline A: 3-cam GCLIP argmax (from eval_20260319_072315.log)
    baseline_sr = 12/30
    baseline_coll = 6/30

    # Parse OACP run log if available
    oacp_log = sorted(glob.glob("log/eval_gclip_oacp_*.log"))
    oacp_sr, oacp_coll = None, None
    if oacp_log:
        _sr, _spl, _c, _ = load_nav_log(oacp_log[-1])
        oacp_sr, oacp_coll = _sr, (_c / 30 if _c is not None else None)

    # Parse fixed-tau log if available
    fixed_log = sorted(glob.glob("log/eval_gclip_fixed_tau_*.log"))
    fixed_sr, fixed_coll = None, None
    if fixed_log:
        _sr, _spl, _c, _ = load_nav_log(fixed_log[-1])
        fixed_sr, fixed_coll = _sr, (_c / 30 if _c is not None else None)

    oacp_coverage = (1 - df_oacp["err"].mean()) if not df_oacp.empty else None

    print_nav_table([
        ("Baseline (3-cam, no CP)", baseline_sr, baseline_coll, None),
        ("GCLIP fixed-τ=0.22", fixed_sr, fixed_coll, None),
        ("GCLIP OACP (Ours)", oacp_sr, oacp_coll, oacp_coverage),
    ])


if __name__ == "__main__":
    main()
