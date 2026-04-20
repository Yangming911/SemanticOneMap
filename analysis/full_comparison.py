"""
Full comparison analysis across all ACP4OneMap conditions.

Generates:
  figures/main_results_table.png    — bar chart: SR / SemColl / STUCK by condition
  figures/safety_frontier.png       — scatter: SR vs Collision rate tradeoff
  figures/tau_convergence_all.png   — τ trajectories for D and F
  figures/coverage_convergence_all.png — coverage convergence D and F
  figures/collision_audit.png       — collision causes pie chart (condition A)

Usage:
  conda run -n onemap python analysis/full_comparison.py
"""

import argparse
import glob
import os
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Per-condition data ─────────────────────────────────────────────────────────
# Parsed from logs and OACP calibration CSVs

def load_oacp_csv(path_pattern):
    files = sorted(glob.glob(os.path.join(BASE, path_pattern)))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_collision_causes(path_pattern):
    """Returns Counter of cause_label → count."""
    files = sorted(glob.glob(os.path.join(BASE, path_pattern)))
    causes = Counter()
    for f in files:
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("cause_label"):
                    causes[row["cause_label"]] += 1
    return causes


def parse_state_files(result_dir):
    """Count outcomes from state files."""
    state_dir = os.path.join(BASE, result_dir, "state")
    if not os.path.exists(state_dir):
        return {}
    counts = Counter()
    for fname in os.listdir(state_dir):
        fpath = os.path.join(state_dir, fname)
        try:
            with open(fpath) as f:
                content = f.read()
                if "SUCCESS" in content:
                    counts["success"] += 1
                elif "SEMANTIC_COLLISION" in content:
                    counts["sem_collision"] += 1
                elif "STUCK" in content:
                    counts["stuck"] += 1
                elif "OOT" in content:
                    counts["oot"] += 1
                elif "NOT_REACHED" in content:
                    counts["not_reached"] += 1
                elif "ALL_EXPLORED" in content:
                    counts["all_explored"] += 1
                elif "MISDETECT" in content:
                    counts["misdetect"] += 1
        except Exception:
            pass
    return counts


def bootstrap_ci(data, stat_fn=np.mean, n=2000, alpha=0.05):
    """Bootstrap 95% CI for a statistic."""
    data = np.array(data)
    samples = np.random.choice(data, size=(n, len(data)), replace=True)
    stats = [stat_fn(s) for s in samples]
    lo = np.percentile(stats, 100 * alpha / 2)
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ── Condition definitions ─────────────────────────────────────────────────────

CONDITIONS = {
    "A": {
        "label": "A: GCLIP argmax\n(no CP)",
        "color": "#4C72B0",
        "n_eps": 30,
        "successes": 12, "sem_coll": 6, "stuck": 0, "not_reached": 2, "oot": 3,
        "misdetect": 6, "all_explored": 1,
        "path_m": 6.7 * 6,   # avg 6.7m × 6 collisions for matched-progress
        "coverage": None,
        "tau_mean": None,
        "oacp_csv": None,
    },
    "C": {
        "label": "C: OACP\ns=1−sim",
        "color": "#DD8452",
        "n_eps": 30,
        "successes": 0, "sem_coll": 4, "stuck": 11, "not_reached": 2, "oot": 0,
        "misdetect": 5, "all_explored": 8,
        "path_m": None,
        "coverage": None,   # τ→0.77 but OACP was s=1-sim
        "tau_mean": 0.77,
        "oacp_csv": "results/gclip_oacp/oacp_calibration_*.csv",
    },
    "D": {
        "label": "D: OACP margin\n90% coverage",
        "color": "#55A868",
        "n_eps": 30,
        "successes": 1, "sem_coll": 2, "stuck": 14, "not_reached": 2, "oot": 0,
        "misdetect": 5, "all_explored": 6,
        "path_m": 1.38,   # coll/100m
        "coverage": 0.894,
        "tau_mean": 0.462,
        "oacp_csv": "results/gclip_oacp_margin/oacp_calibration_*.csv",
    },
    "F": {
        "label": "F: OACP margin\n70% coverage",
        "color": "#C44E52",
        "n_eps": 30,  # final
        "successes": 1, "sem_coll": 2, "stuck": 7, "not_reached": 2, "oot": 0,
        "misdetect": 7, "all_explored": 11,
        "path_m": None,
        "coverage": 0.700,
        "tau_mean": 0.468,
        "oacp_csv": "results/gclip_oacp_margin70/oacp_calibration_*.csv",
    },
}

# ── Figure 1: Main results bar chart ─────────────────────────────────────────

def plot_main_results(conditions, out_path):
    names = list(conditions.keys())
    labels = [conditions[k]["label"] for k in names]
    n_eps = [conditions[k]["n_eps"] for k in names]

    sr = [conditions[k]["successes"] / conditions[k]["n_eps"] * 100 for k in names]
    coll = [conditions[k]["sem_coll"] / conditions[k]["n_eps"] * 100 for k in names]
    stuck = [conditions[k]["stuck"] / conditions[k]["n_eps"] * 100 for k in names]

    x = np.arange(len(names))
    w = 0.25
    colors = [conditions[k]["color"] for k in names]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars_sr = ax.bar(x - w, sr, w, label="Success Rate ↑", color=[c + "CC" for c in colors])
    bars_coll = ax.bar(x, coll, w, label="Sem. Collision ↓", color=colors)
    bars_stuck = ax.bar(x + w, stuck, w, label="STUCK ↓", color=[c + "77" for c in colors])

    # Annotate bars
    for bar, val in zip(bars_sr, sr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_coll, coll):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_stuck, stuck):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title("Navigation Results by Condition (HM3D val-mini)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 80)
    ax.grid(True, axis="y", alpha=0.3)

    # Add n_eps annotation
    for i, n in enumerate(n_eps):
        ax.text(x[i], 76, f"n={n}", ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: Safety-efficiency frontier ──────────────────────────────────────

def plot_frontier(conditions, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))

    for k, cond in conditions.items():
        n = cond["n_eps"]
        sr = cond["successes"] / n * 100
        coll = cond["sem_coll"] / n * 100
        color = cond["color"]
        label = cond["label"].replace("\n", " ")

        ax.scatter(coll, sr, color=color, s=120, zorder=5)
        ax.annotate(label, (coll, sr),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=color)

    # Draw ideal direction arrow
    ax.annotate("", xy=(0, 40), xytext=(5, 40),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.text(0.5, 38, "safer →", fontsize=8, color="gray")
    ax.annotate("", xy=(0, 42), xytext=(0, 37),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.text(1, 40, "↑ more mobile", fontsize=8, color="gray")

    ax.set_xlabel("Semantic Collision Rate (%)", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_title("Safety-Efficiency Frontier\n(lower-left = safer, upper-right = more mobile)",
                 fontsize=11)
    ax.set_xlim(-1, 25)
    ax.set_ylim(-2, 50)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 3: τ convergence (D + F) ───────────────────────────────────────────

def plot_tau_all(conditions, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key in zip(axes, ["D", "F"]):
        cond = conditions[key]
        if not cond["oacp_csv"]:
            continue
        df = load_oacp_csv(cond["oacp_csv"])
        if df.empty:
            ax.set_title(f"{key}: no data")
            continue
        for ep_id, grp in df.groupby("episode_id"):
            ax.plot(grp["step"].values, grp["tau"].values,
                    color=cond["color"], alpha=0.2, linewidth=0.7)
        mean_tau = df.groupby("step")["tau"].mean()
        ax.plot(mean_tau.index, mean_tau.values,
                color=cond["color"], linewidth=2.5,
                label=f"Mean τ (final={df.groupby('episode_id')['tau'].last().mean():.3f})")
        ax.axhline(cond["tau_mean"] or 0.5, color="red", linestyle="--", linewidth=1, alpha=0.7,
                   label=f"τ_target≈{cond['tau_mean']:.3f}")
        ax.set_xlabel("Calibration step t", fontsize=10)
        ax.set_ylabel("τ_t", fontsize=10)
        ax.set_title(f"Condition {key}: τ Convergence\n(coverage target={cond['coverage']:.0%})", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        n_eps = df["episode_id"].nunique()
        coverage = 1 - df["err"].mean()
        ax.text(0.97, 0.05, f"n_eps={n_eps}\ncoverage={coverage:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    fig.suptitle("ACI Threshold Convergence — Theorem 1", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 4: Coverage convergence (D + F) ────────────────────────────────────

def plot_coverage_all(conditions, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, key in zip(axes, ["D", "F"]):
        cond = conditions[key]
        if not cond["oacp_csv"]:
            continue
        df = load_oacp_csv(cond["oacp_csv"])
        if df.empty:
            ax.set_title(f"{key}: no data")
            continue
        alpha_target = 1.0 - cond["coverage"]
        coverage_target = cond["coverage"]

        # Global running coverage
        all_errs = []
        for _, grp in df.groupby("episode_id"):
            all_errs.extend(grp["err"].tolist())
        running_cov = 1 - np.cumsum(all_errs) / (np.arange(len(all_errs)) + 1)
        ax.plot(np.arange(len(running_cov)), running_cov,
                color=cond["color"], linewidth=2.5, label="Running coverage")
        ax.axhline(coverage_target, color="red", linestyle="--", linewidth=1.5,
                   label=f"Target {coverage_target:.0%}")
        ax.set_xlabel("Cumulative calibration step", fontsize=10)
        ax.set_ylabel("Coverage rate", fontsize=10)
        ax.set_title(f"Condition {key}: Coverage Convergence", fontsize=10)
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        final_cov = 1 - df["err"].mean()
        ax.text(0.97, 0.05, f"Final coverage={final_cov:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    fig.suptitle("ACI Coverage Convergence — Theorem 1", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 5: Collision causes audit ──────────────────────────────────────────

def plot_collision_audit(out_path):
    causes_A = load_collision_causes("results/gclip_argmax_3cam/collision_causes_*.csv")
    if not causes_A:
        print("  No collision causes data for condition A")
        return

    # Filter to non-zero
    items = sorted(causes_A.items(), key=lambda x: -x[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    colors_pie = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        vals, labels=labels, autopct="%1.0f%%",
        colors=colors_pie, startangle=90,
        textprops=dict(fontsize=10)
    )
    ax.set_title(f"Condition A — Semantic Collision Causes\n(n={sum(vals)} total collisions)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Text summary ───────────────────────────────────────────────────────────────

def print_full_summary(conditions):
    print("\n" + "="*72)
    print("ACP4OneMap — Full Results Summary")
    print("="*72)
    print(f"{'Cond':<6} {'n':>4} {'SR':>8} {'SemColl':>10} {'STUCK':>8} {'Coverage':>10} {'τ_mean':>8}")
    print("-"*72)
    for k, cond in conditions.items():
        n = cond["n_eps"]
        sr = cond["successes"] / n
        coll = cond["sem_coll"] / n
        stuck = cond["stuck"] / n
        cov = f"{cond['coverage']:.3f}" if cond["coverage"] is not None else "N/A"
        tau = f"{cond['tau_mean']:.3f}" if cond["tau_mean"] is not None else "N/A"
        interim = "*" if k == "F" else ""
        print(f"{k+interim:<6} {n:>4} {sr:>8.1%} {coll:>10.1%} {stuck:>8.1%} {cov:>10} {tau:>8}")
    print("* = interim (F not yet 30 eps)")

    # Bootstrap CIs for A and D
    print("\nBootstrap 95% CIs (n=30 each):")
    for k, cond in conditions.items():
        if cond["n_eps"] < 30:
            continue
        n = cond["n_eps"]
        sr_vec = [1] * cond["successes"] + [0] * (n - cond["successes"])
        coll_vec = [1] * cond["sem_coll"] + [0] * (n - cond["sem_coll"])
        sr_lo, sr_hi = bootstrap_ci(sr_vec)
        coll_lo, coll_hi = bootstrap_ci(coll_vec)
        print(f"  {k}: SR={cond['successes']/n:.0%} [{sr_lo:.0%}, {sr_hi:.0%}]  "
              f"Coll={cond['sem_coll']/n:.0%} [{coll_lo:.0%}, {coll_hi:.0%}]")
    print("="*72)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Generating full comparison analysis...")

    # Run figures
    plot_main_results(CONDITIONS, os.path.join(FIG_DIR, "main_results_table.png"))
    plot_frontier(CONDITIONS, os.path.join(FIG_DIR, "safety_frontier.png"))
    plot_tau_all(CONDITIONS, os.path.join(FIG_DIR, "tau_convergence_all.png"))
    plot_coverage_all(CONDITIONS, os.path.join(FIG_DIR, "coverage_convergence_all.png"))
    plot_collision_audit(os.path.join(FIG_DIR, "collision_audit.png"))
    print_full_summary(CONDITIONS)

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
