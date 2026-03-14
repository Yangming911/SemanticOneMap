"""
SemanticPredGTCollector: accumulates (CLIP-predicted label, GT label) pairs
and max-similarity score distributions across the full evaluation run.

Outputs two CSVs:
  - semantic_pred_gt_pairs.csv   : (pred, gt) count table
  - semantic_sim_distribution.csv: per-GT-category histogram of max_sim scores
"""
import csv
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np


# Bin edges for max_sim histogram (cosine similarity range roughly [-0.3, 0.5])
_SIM_BINS = np.arange(-0.3, 0.55, 0.025)


class SemanticPredGTCollector:
    def __init__(self) -> None:
        self.counts: Counter = Counter()                         # (pred_label, gt_label) -> count
        # gt_category -> histogram counts over _SIM_BINS
        self.sim_hist: dict = defaultdict(lambda: np.zeros(len(_SIM_BINS) - 1, dtype=np.int64))
        self.gt_seed_map: Optional[np.ndarray] = None
        self.gt_labels: List[str] = []

    def set_gt_map(self, gt_seed_map: np.ndarray, gt_labels: List[str]) -> None:
        self.gt_seed_map = gt_seed_map
        self.gt_labels = gt_labels

    def record(
        self,
        valid_mask: np.ndarray,
        pred_ids: np.ndarray,
        pred_labels: List[str],
        max_sims: Optional[np.ndarray] = None,  # 1-D float array, same length as pred_ids
    ) -> None:
        if self.gt_seed_map is None or valid_mask.sum() == 0:
            return

        gt_at_cells = self.gt_seed_map[valid_mask]

        for i, (pred_idx, gt_idx) in enumerate(zip(pred_ids, gt_at_cells)):
            pred_label = (
                pred_labels[int(pred_idx) - 1]
                if 1 <= int(pred_idx) <= len(pred_labels)
                else "unknown"
            )
            gt_label = (
                self.gt_labels[int(gt_idx) - 1]
                if int(gt_idx) > 0 and int(gt_idx) <= len(self.gt_labels)
                else "free"
            )
            self.counts[(pred_label, gt_label)] += 1

            if max_sims is not None:
                sim_val = float(max_sims[i])
                bin_idx = np.searchsorted(_SIM_BINS, sim_val, side="right") - 1
                bin_idx = int(np.clip(bin_idx, 0, len(_SIM_BINS) - 2))
                self.sim_hist[gt_label][bin_idx] += 1

    def save_csv(self, path: str) -> None:
        total = sum(self.counts.values())
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pred_label", "gt_label", "count", "pct"])
            for (pred, gt), count in sorted(self.counts.items(), key=lambda x: -x[1]):
                writer.writerow([pred, gt, count, f"{count / total * 100:.2f}"])
        print(f"[SemanticDebug] Saved {len(self.counts)} pred-GT pairs ({total} cells) → {path}")

    def save_sim_distribution_csv(self, path: str) -> None:
        bin_labels = [f"{_SIM_BINS[i]:.3f}~{_SIM_BINS[i+1]:.3f}" for i in range(len(_SIM_BINS) - 1)]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gt_label", "bin", "bin_center", "count"])
            for gt_label, hist in sorted(self.sim_hist.items()):
                total_cat = hist.sum()
                for i, count in enumerate(hist):
                    if count == 0:
                        continue
                    bin_center = round(float((_SIM_BINS[i] + _SIM_BINS[i + 1]) / 2), 4)
                    writer.writerow([gt_label, bin_labels[i], bin_center, int(count)])
        print(f"[SemanticDebug] Saved sim distribution ({len(self.sim_hist)} categories) → {path}")