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
        window_size: int = 200,
        gamma: float = 0.05,
        initial_alpha: float = 0.5,
        use_margin_score: bool = False,
        max_seeds_per_label: int = 0,   # 0 = unlimited; >0 = keep only top-K seeds per label
        temporal_persistence: int = 0,  # 0 = disabled; N>0 = cell must be seed for N consecutive frames
    ) -> None:
        self.n_cells = n_cells
        self.cell_size = cell_size
        self.threshold = threshold          # τ in similarity space
        self.use_oacp = use_oacp
        self.target_coverage = target_coverage

        # Margin-based score mode: s = (max_bg_sim - sim(feat, label) + 1) / 2
        # Replaces raw 1-sim score; threshold lives near 0.5 (argmax boundary)
        self.use_margin_score = use_margin_score
        self.max_seeds_per_label = int(max_seeds_per_label)
        self.temporal_persistence = int(temporal_persistence)

        # ACI state
        self._initial_alpha = initial_alpha  # stored for reset()
        self._initial_threshold = threshold  # stored for reset()
        self._alpha_t = initial_alpha        # risk level (miscoverage rate)
        self._gamma = gamma                  # ACI step size
        self._alpha_target = 1.0 - target_coverage  # target miscoverage rate

        # Set by set_text_features() after CLIP model is ready
        self._text_features: Optional[torch.Tensor] = None  # [N_obs, F]
        self._labels: List[str] = []
        self._radii: List[int] = []
        self._kernels: List[Optional[np.ndarray]] = []

        # Optional background/non-obstacle reference labels for argmax mode
        self._bg_text_features: Optional[torch.Tensor] = None  # [N_bg, F]
        self._bg_labels: List[str] = []  # names for each bg label

        # Per-cell state: 0 = free, label_idx+1 = predicted obstacle
        self._seed_map = np.zeros((n_cells, n_cells), dtype=np.uint8)
        # Temporal persistence: remember raw seeds from previous frame
        self._prev_seed_map = np.zeros((n_cells, n_cells), dtype=np.uint8)
        # Nonconformity score at each seed cell (lower = more confident obstacle)
        self._seed_score_map = np.ones((n_cells, n_cells), dtype=np.float32)
        # For bg cells: winning bg label index+1 (0=unobserved or obstacle winner)
        self._bg_winner_map = np.zeros((n_cells, n_cells), dtype=np.uint16)
        self._obstacle_mask = np.zeros((n_cells, n_cells), dtype=bool)

        # OACP calibration buffer (nonconformity scores)
        self._scores: deque = deque(maxlen=window_size)

        # Per-call calibration log for experiment analysis
        self._calib_step: int = 0
        self._calibration_log: List[dict] = []

    # ------------------------------------------------------------------
    def set_text_features(
        self,
        text_features: torch.Tensor,
        labels: List[str],
        radii: List[int],
        bg_text_features: Optional[torch.Tensor] = None,
        bg_labels: Optional[List[str]] = None,
    ) -> None:
        """Called once at init with precomputed obstacle label embeddings.

        Args:
            bg_text_features: [N_bg, F] background/non-obstacle label embeddings.
                When provided, uses argmax mode: a cell is an obstacle only if
                its best obstacle-label sim beats all background-label sims.
                When None, falls back to threshold mode (sim >= self.threshold).
        """
        self._text_features = text_features   # [N_obs, F], already normalised
        self._labels = labels
        self._radii = radii
        self._kernels = [self._make_disk(r) for r in radii]
        self._bg_text_features = bg_text_features  # [N_bg, F] or None
        self._bg_labels = bg_labels if bg_labels is not None else []

    @staticmethod
    def _make_disk(radius: int) -> Optional[np.ndarray]:
        if radius <= 0:
            return np.ones((1, 1), dtype=np.uint8)
        d = radius * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    @property
    def argmax_mode(self) -> bool:
        return self._bg_text_features is not None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._seed_map.fill(0)
        self._seed_score_map.fill(1.0)
        self._obstacle_mask.fill(False)
        self._bg_winner_map.fill(0)
        self._prev_seed_map.fill(0)
        self._scores.clear()
        self._alpha_t = self._initial_alpha    # restore configured initial value
        self.threshold = self._initial_threshold
        self._calib_step = 0
        self._calibration_log = []

    def get_calibration_log(self) -> List[dict]:
        """Return a copy of the calibration log and clear it."""
        log = list(self._calibration_log)
        self._calibration_log = []
        self._calib_step = 0
        return log

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
    def calibrate_aci(
        self,
        clip_feat: torch.Tensor,
        true_label: str,
    ) -> None:
        """ACI update using GT label (real-time, no distance binning).

        Called when the robot is within δ_obs of a cell whose GT label is known.
        Adjusts α_t via gradient step so that coverage tracks α_target over time,
        then updates τ from the calibration score buffer.

        Args:
            clip_feat: CLIP feature vector at the cell [F].
            true_label: ground-truth semantic label of that cell.
        """
        if not self.use_oacp or self._text_features is None:
            return
        if true_label not in self._labels:
            return
        feat_norm = clip_feat.norm()
        if feat_norm < 1e-6:
            return  # unobserved cell

        j = self._labels.index(true_label)
        text_feat = self._text_features[j].to(clip_feat.device)
        feat_n = F.normalize(clip_feat.unsqueeze(0), dim=1)
        sim = float(feat_n.mm(text_feat.unsqueeze(0).T).squeeze())

        if self.use_margin_score and self._bg_text_features is not None:
            # Margin score: s = (max_bg_sim - sim + 1) / 2 ∈ [0, 1]
            # Obstacle cells: sim > max_bg → s < 0.5 (conformal)
            # BG cells:       max_bg > sim → s > 0.5 (not conformal)
            bg_feats = self._bg_text_features.to(clip_feat.device)
            if bg_feats.dtype != feat_n.dtype:
                bg_feats = bg_feats.to(feat_n.dtype)
            max_bg_sim = float((feat_n @ bg_feats.T).squeeze(0).max())
            s = (max_bg_sim - sim + 1.0) / 2.0
        else:
            s = 1.0 - sim  # nonconformity score: low = good match

        # err = 1 if current threshold does NOT cover the true label
        C_t = 1.0 - self.threshold  # nonconformity threshold
        err = float(s > C_t)
        tau_before = self.threshold  # log pre-update tau for correct convergence plots

        # ACI gradient step
        self._alpha_t = float(np.clip(
            self._alpha_t + self._gamma * (self._alpha_target - err),
            1e-4, 1.0 - 1e-4,
        ))

        # Update τ: nonconformity threshold = (1-α_t) quantile of stored scores
        self._scores.append(s)
        if len(self._scores) >= 5:
            C_new = float(np.quantile(list(self._scores), 1.0 - self._alpha_t))
            self.threshold = float(np.clip(1.0 - C_new, 0.0, 1.0))

        # Log calibration call: tau_before pairs with err at this step
        self._calibration_log.append({
            "step": self._calib_step,
            "tau": tau_before,
            "tau_after": self.threshold,
            "err": err,
            "alpha": self._alpha_t,
            "s": s,
        })
        self._calib_step += 1

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

        feats_norm = F.normalize(feats, dim=1)        # [M, F]
        obs_sims = (feats_norm @ text_feats.T).detach().cpu().numpy()   # [M, N_obs]
        norms_np = feat_norms.detach().cpu().numpy()

        if self._bg_text_features is not None:
            bg_feats = self._bg_text_features.to(feats.device)
            if bg_feats.dtype != feats.dtype:
                bg_feats = bg_feats.to(feats.dtype)
            bg_sims = (feats_norm @ bg_feats.T).detach().cpu().numpy()  # [M, N_bg]
            best_bg_sim = bg_sims.max(axis=1)                            # [M]
        else:
            best_bg_sim = None

        xs, ys = np.nonzero(changed_np)
        new_seed = self._seed_map.copy()
        new_bg_winner = self._bg_winner_map.copy()

        # Vectorized cell classification (replaces per-cell Python loop)
        valid = norms_np >= 1e-6                               # [M] bool
        best_j = np.argmax(obs_sims, axis=1)                  # [M]
        best_sim = obs_sims[np.arange(len(obs_sims)), best_j] # [M]

        new_seed[xs[~valid], ys[~valid]] = 0

        new_score = self._seed_score_map.copy()

        if valid.any():
            vxs, vys = xs[valid], ys[valid]
            vbest_j   = best_j[valid]
            vbest_sim = best_sim[valid]

            if self.use_margin_score and best_bg_sim is not None:
                # Margin-OACP: s = (max_bg - sim + 1) / 2 <= 1 - tau  →  obstacle
                vs = (best_bg_sim[valid] - vbest_sim + 1.0) / 2.0
                is_obs = vs <= (1.0 - self.threshold)
                new_seed[vxs[is_obs],  vys[is_obs]]  = vbest_j[is_obs] + 1
                new_score[vxs[is_obs],  vys[is_obs]]  = vs[is_obs]
                new_score[vxs[~is_obs], vys[~is_obs]] = 1.0
                new_seed[vxs[~is_obs], vys[~is_obs]] = 0
                new_bg_winner[vxs[is_obs],  vys[is_obs]]  = 0
                new_bg_winner[vxs[~is_obs], vys[~is_obs]] = (
                    np.argmax(bg_sims[valid][~is_obs], axis=1) + 1
                )
            elif best_bg_sim is not None:
                # Argmax mode: obstacle wins only if it beats all background labels
                is_obs = vbest_sim > best_bg_sim[valid]
                new_seed[vxs[is_obs],  vys[is_obs]]  = vbest_j[is_obs] + 1
                new_seed[vxs[~is_obs], vys[~is_obs]] = 0
                new_bg_winner[vxs[is_obs],  vys[is_obs]]  = 0
                new_bg_winner[vxs[~is_obs], vys[~is_obs]] = (
                    np.argmax(bg_sims[valid][~is_obs], axis=1) + 1
                )
            else:
                # Threshold mode: obstacle wins if sim >= τ
                is_obs = vbest_sim >= self.threshold
                new_seed[vxs[is_obs],  vys[is_obs]]  = vbest_j[is_obs] + 1
                new_seed[vxs[~is_obs], vys[~is_obs]] = 0

        # Temporal persistence: only keep seeds that were also seeds last frame
        if self.temporal_persistence > 0:
            stable = (new_seed > 0) & (self._prev_seed_map > 0)
            new_seed[~stable] = 0
            new_score[~stable] = 1.0
        self._prev_seed_map = new_seed.copy()  # remember raw seeds before TopK for next frame

        if not np.array_equal(new_seed, self._seed_map):
            self._seed_map = new_seed
            self._seed_score_map = new_score
            self._recompute_obstacle_mask()
        self._bg_winner_map = new_bg_winner

    # ------------------------------------------------------------------
    def _recompute_obstacle_mask(self) -> None:
        combined = np.zeros((self.n_cells, self.n_cells), dtype=bool)
        for j, kernel in enumerate(self._kernels):
            if kernel is None:
                continue
            seed = (self._seed_map == j + 1)
            if not seed.any():
                continue
            # TopK: keep only the max_seeds_per_label cells with lowest score (most confident)
            if self.max_seeds_per_label > 0:
                seed_xs, seed_ys = np.nonzero(seed)
                n_seeds = len(seed_xs)
                if n_seeds > self.max_seeds_per_label:
                    scores = self._seed_score_map[seed_xs, seed_ys]
                    topk_idx = np.argpartition(scores, self.max_seeds_per_label)[:self.max_seeds_per_label]
                    seed_pruned = np.zeros_like(seed)
                    seed_pruned[seed_xs[topk_idx], seed_ys[topk_idx]] = True
                    seed = seed_pruned
            combined |= cv2.dilate(seed.astype(np.uint8), kernel, iterations=1).astype(bool)
        self._obstacle_mask = combined

    # ------------------------------------------------------------------
    def get_obstacle_mask(self) -> np.ndarray:
        return self._obstacle_mask

    def apply_to_navigable_map(self, base: np.ndarray) -> np.ndarray:
        return base & ~self._obstacle_mask
