"""Combine 3 shard log files into a single summary for val_300 experiments."""
import re
import sys
from pathlib import Path

LOG_DIR = Path("/opt/data/private/onemap/OneMap_obstacle/log")

RESULT_LABELS = [
    "SUCCESS", "FAILURE_MISDETECT", "FAILURE_STUCK",
    "FAILURE_OOT", "FAILURE_NOT_REACHED", "FAILURE_ALL_EXPLORED",
    "SEMANTIC_COLLISION",
]

def parse_log(path):
    text = Path(path).read_text()
    counts = {}
    for label in RESULT_LABELS:
        m = re.search(rf"{label}\s+(\d+)\s*/\s*(\d+)", text)
        if m:
            counts[label] = (int(m.group(1)), int(m.group(2)))
    spl_m = re.search(r"SPL\s*:\s*([\d.]+)", text)
    spl = float(spl_m.group(1)) if spl_m else None
    path_m = re.search(r"Avg path length\s*:\s*([\d.]+)", text)
    avg_path = float(path_m.group(1)) if path_m else None
    return counts, spl, avg_path


def combine(log_paths, label):
    total_counts = {k: [0, 0] for k in RESULT_LABELS}
    spl_sum, spl_n = 0.0, 0
    path_sum, path_n = 0.0, 0

    for p in log_paths:
        counts, spl, avg_path = parse_log(p)
        for k in RESULT_LABELS:
            if k in counts:
                total_counts[k][0] += counts[k][0]
                total_counts[k][1] += counts[k][1]
        if spl is not None:
            n = total_counts["SUCCESS"][1] if total_counts["SUCCESS"][1] else 1
            spl_sum += spl
            spl_n += 1
        if avg_path is not None:
            path_sum += avg_path
            path_n += 1

    n_eps = total_counts["SUCCESS"][1]
    success = total_counts["SUCCESS"][0]
    sr = success / n_eps if n_eps else 0
    avg_spl = spl_sum / spl_n if spl_n else 0
    avg_path = path_sum / path_n if path_n else 0

    print(f"\n{'='*60}")
    print(f"  COMBINED: {label}  ({n_eps} episodes)")
    print(f"{'='*60}")
    print(f"  SR  : {sr:.1%}  ({success}/{n_eps})")
    print(f"  SPL : {avg_spl:.3f}  (avg of shards)")
    print(f"  Avg path length: {avg_path:.1f} m")
    print(f"\n  Breakdown:")
    for k in RESULT_LABELS:
        c, t = total_counts[k]
        if t > 0:
            print(f"    {k:<30} {c:3d} / {t:3d}  ({c/t:.1%})")


if __name__ == "__main__":
    import glob

    # Find latest shard logs automatically
    for experiment in ("baseline", "yolo"):
        logs = []
        for s in (0, 1, 2):
            pattern = str(LOG_DIR / f"val300_{experiment}_s{s}_*.log")
            matches = sorted(glob.glob(pattern))
            if matches:
                logs.append(matches[-1])  # latest
                print(f"  shard {s}: {Path(matches[-1]).name}")
        if len(logs) == 3:
            combine(logs, f"val_300 {experiment}")
        else:
            print(f"\n[{experiment}] only {len(logs)}/3 shard logs found, skipping")
