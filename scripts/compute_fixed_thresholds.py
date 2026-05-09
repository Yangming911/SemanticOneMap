"""Compute fixed CP thresholds from offline calibration scores.

Usage:
    python scripts/compute_fixed_thresholds.py results/mp3d_calibration/oacp_calibration_*.csv
"""
import sys
import glob
import numpy as np

def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "results/mp3d_calibration/oacp_calibration_*.csv"
    files = glob.glob(pattern)
    if not files:
        print(f"No files matching: {pattern}")
        sys.exit(1)

    scores = []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(",")
            s_idx = header.index("s")
            for line in fh:
                vals = line.strip().split(",")
                scores.append(float(vals[s_idx]))

    scores = np.array(scores)
    print(f"Loaded {len(scores)} nonconformity scores from {len(files)} file(s)")
    print(f"  min={scores.min():.4f}  max={scores.max():.4f}  mean={scores.mean():.4f}  std={scores.std():.4f}")
    print()

    coverages = [0.70, 0.80, 0.90, 0.95]
    print("Coverage  →  quantile(s, cov)  →  τ = 1 - quantile")
    print("-" * 55)
    for cov in coverages:
        q = np.quantile(scores, cov)
        tau = 1.0 - q
        print(f"  {cov:.2f}     →  {q:.4f}            →  τ = {tau:.4f}")

if __name__ == "__main__":
    main()
