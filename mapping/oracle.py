from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class OracleDetection:
    label: str
    cell_x: int
    cell_y: int


class ObstacleOracle(ABC):
    """Interface for obstacle detection oracles used by OACP.

    An oracle observes the environment each step and returns per-cell
    obstacle detections.  It also provides the initial safety radius
    for newly discovered obstacle categories.

    Implementations:
      - GTLabelMapOracle  : simulation GT semantic map (current default)
      - (future) GroundingDINOOracle, VLMOracle, ...
    """

    @abstractmethod
    def detect(
        self,
        agent_cell: Tuple[int, int],
    ) -> List[OracleDetection]:
        """Return obstacle detections visible from the agent's current position.

        Args:
            agent_cell: (px, py) agent position in map cell coordinates.

        Returns:
            List of OracleDetection(label, cell_x, cell_y).
        """
        ...

    @abstractmethod
    def get_safety_radius(self, label: str) -> Optional[float]:
        """Return the raw safety radius (in cells) for a label, or None if unknown."""
        ...

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass


class GTLabelMapOracle(ObstacleOracle):
    """Oracle backed by a ground-truth semantic label map (simulation)."""

    def __init__(
        self,
        safety_dict: dict,
        observe_radius_m: float = 3.0,
        cell_size: float = 0.1,
    ):
        self._safety_dict = safety_dict
        self._observe_radius_m = observe_radius_m
        self._cell_size = cell_size
        self._gt_label_map: Optional[np.ndarray] = None
        self._gt_labels: List[str] = []

    def set_gt_label_map(self, label_map: np.ndarray, labels: List[str]) -> None:
        self._gt_label_map = label_map
        self._gt_labels = labels

    def detect(
        self,
        agent_cell: Tuple[int, int],
    ) -> List[OracleDetection]:
        if self._gt_label_map is None:
            return []

        px, py = agent_cell
        n_cells = self._gt_label_map.shape[0]
        delta = max(1, int(self._observe_radius_m / self._cell_size))
        x0 = max(0, px - delta)
        x1 = min(n_cells, px + delta + 1)
        y0 = max(0, py - delta)
        y1 = min(n_cells, py + delta + 1)

        region = self._gt_label_map[x0:x1, y0:y1]
        if not region.any():
            return []

        detections = []
        xs, ys = np.nonzero(region)
        for dx, dy in zip(xs, ys):
            lbl_idx = int(region[dx, dy])
            true_label = self._gt_labels[lbl_idx - 1]
            detections.append(OracleDetection(
                label=true_label,
                cell_x=x0 + dx,
                cell_y=y0 + dy,
            ))
        return detections

    def get_safety_radius(self, label: str) -> Optional[float]:
        raw_r = self._safety_dict.get(label)
        if raw_r is None:
            return None
        return raw_r

    def reset(self) -> None:
        pass
