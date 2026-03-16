"""
CLIP-based obstacle map with Conformal Prediction threshold.

TODO 4: Fixed threshold τ — for each CLIP-updated cell, check if
        sim(CLIP_feat, obstacle_label) ≥ τ; if so, mark as obstacle and dilate.

TODO 5: Online ACP — YOLO detections serve as the calibration oracle.
        When YOLO projects an obstacle to (px, py), record the nonconformity
        score = 1 - sim(CLIP_feat[px,py], obstacle_label) in a sliding window.
        τ is updated to maintain target_coverage of those calibration events.

Unexplored cells (feature_norm ≈ 0) are ignored — no prior is applied here.
The planner must separately handle unexplored regions.
"""

from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class CLIPCPObstacleMap:
    def __init__(
        self,
        n_cells: int,
        cell_size: float,
        threshold: float = 0.0,
        use_oacp: bool = False,
        target_coverage: float = 0.9,
        window_size: int = 100,
    ) -> None:
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.threshold = threshold          # τ: fixed (TODO4) or adaptive (TODO5)
        self.use_oacp = use_oacp
        self.target_coverage = target_coverage

        # Set by set_text_features() after CLIP model is ready
        self._text_features: Optional[torch.Tensor] = None  # [N, F]
        self._labels: List[str] = []
        self._radii: List[int] = []
        self._kernels: List[Optional[np.ndarray]] = []

        # Per-cell state: 0 = free, label_idx+1 = predicted obstacle
        self._seed_map = np.zeros((n_cells, n_cells), dtype=np.uint8)
        self._obstacle_mask = np.zeros((n_cells, n_cells), dtype=bool)

        # OACP calibration buffer
        self._scores: deque = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    def set_text_features(
        self,
        text_features: torch.Tensor,
        labels: List[str],
        radii: List[int],
    ) -> None:
        """Called once at init with precomputed obstacle label embeddings."""
        self._text_features = text_features   # [N, F], already normalised
        self._labels = labels
        self._radii = radii
        self._kernels = [self._make_disk(r) for r in radii]

    @staticmethod
    def _make_disk(radius: int) -> Optional[np.ndarray]:
        if radius <= 0:
            return np.ones((1, 1), dtype=np.uint8)
        d = radius * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._seed_map.fill(0)
        self._obstacle_mask.fill(False)
        self._scores.clear()

    # ------------------------------------------------------------------
    def calibrate(
        self,
        clip_feat: torch.Tensor,
        yolo_label: str,
    ) -> None:
        """OACP (TODO 5): add a calibration point from a YOLO detection.

        Args:
            clip_feat: CLIP feature vector at the projected cell [F].
            yolo_label: the obstacle label YOLO detected at that cell.
        """
        if not self.use_oacp or self._text_features is None:
            return
        if yolo_label not in self._labels:
            return
        norm = clip_feat.norm()
        if norm < 1e-6:
            return  # unexplored cell — skip, no calibration signal
        j = self._labels.index(yolo_label)
        text_feat = self._text_features[j].to(clip_feat.device)
        sim = float(F.normalize(clip_feat.unsqueeze(0), dim=1)
                    .mm(text_feat.unsqueeze(0).T)   # text_feat already normalised
                    .squeeze())
        score = 1.0 - sim   # nonconformity score: high when CLIP missed the obstacle
        self._scores.append(score)
        if len(self._scores) >= 10:
            # τ = 1 - (target_coverage quantile of nonconformity scores)
            self.threshold = float(1.0 - np.quantile(list(self._scores), self.target_coverage))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        updated_mask: torch.Tensor,   # bool mask [X, Y] of cells with new CLIP features
        feature_map: torch.Tensor,    # [X, Y, F]
    ) -> None:
        """Recompute obstacle seeds for all CLIP-updated cells."""
        if self._text_features is None:
            return
        if not updated_mask.any():
            return

        changed_np = updated_mask.detach().cpu().numpy().astype(bool)
        feats = feature_map[updated_mask, :]          # [M, F]
        feat_norms = feats.norm(dim=1)               # [M]

        text_feats = self._text_features.to(feats.device)
        if text_feats.dtype != feats.dtype:
            text_feats = text_feats.to(feats.dtype)

        feats_norm = F.normalize(feats, dim=1)        # cosine similarity requires normalised feats
        sims = feats_norm @ text_feats.T             # [M, N_labels], range [-1, 1]
        sims_np = sims.detach().cpu().numpy()
        norms_np = feat_norms.detach().cpu().numpy()

        xs, ys = np.nonzero(changed_np)
        new_seed = self._seed_map.copy()

        for i, (cx, cy) in enumerate(zip(xs, ys)):
            if norms_np[i] < 1e-6:
                # Unexplored cell — clear any stale seed
                new_seed[cx, cy] = 0
                continue
            # Prediction set: labels with sim ≥ τ
            # Pick the highest-similarity obstacle label that exceeds τ
            best_j = -1
            best_sim = -1.0
            for j in range(len(self._labels)):
                if sims_np[i, j] >= self.threshold and sims_np[i, j] > best_sim:
                    best_sim = sims_np[i, j]
                    best_j = j
            new_seed[cx, cy] = (best_j + 1) if best_j >= 0 else 0

        if not np.array_equal(new_seed, self._seed_map):
            self._seed_map = new_seed
            self._recompute_obstacle_mask()

    # ------------------------------------------------------------------
    def _recompute_obstacle_mask(self) -> None:
        combined = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        for j, kernel in enumerate(self._kernels):
            if kernel is None:
                continue
            seed = (self._seed_map == j + 1).astype(np.uint8)
            if seed.max() == 0:
                continue
            combined |= cv2.dilate(seed, kernel, iterations=1).astype(bool)
        self._obstacle_mask = combined

    # ------------------------------------------------------------------
    def get_obstacle_mask(self) -> np.ndarray:
        return self._obstacle_mask

    def apply_to_navigable_map(self, base: np.ndarray) -> np.ndarray:
        return base & ~self._obstacle_mask
