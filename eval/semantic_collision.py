from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


_SEMANTIC_LABEL_ALIASES = {
    "tv monitor": "tv",
    "tvmonitor": "tv",
    "television": "tv",
    "monitor": "tv",
    "plant": "potted plant",
    "potted plant": "potted plant",
    "diningtable": "dining table",
    "coffee table": "dining table",
    "sofa": "couch",
}

_SEMANTIC_SAFETY_RADIUS_CELLS = {
    "person": 3,
    "chair": 2,
    "couch": 3,
    "bed": 2,
    "dining table": 3,
    "potted plant": 2,
    "tv": 2,
    "toilet": 0,
    "sink": 2,
    "oven": 2,
    "refrigerator": 3,
    "microwave": 2,
    "bottle": 2,
    "cup": 2,
    "vase": 2,
    "clock": 1,
    "book": 1,
    "laptop": 1,
}

_DEFAULT_SAFETY_RADIUS_CELLS = 1

_NON_OBSTACLE_LABELS = frozenset({
    # 建筑结构
    "unknown", "ceiling", "floor", "wall", "void", "misc",
    "window", "curtain", "stairs", "staircase",
    "door", "dorr", "door frame", "pillar", "balustrade",
    # 天花板/高处悬挂，机器人不会碰到
    "chandelier", "light", "ventilation", "fire alarm",
    # 贴墙的小型控制面板/装饰
    "handle", "alarm control", "shower dial", "electrical controller",
    "curtain rod",
    # 墙面装饰/挂件
    "photo", "picture", "frame", "flag", "decorative plate",
    # 桌面/台面杂物类（太模糊）
    "kitchen countertop item",
})


@dataclass(frozen=True)
class SemanticCollisionData:
    collision_map: np.ndarray
    label_map: np.ndarray
    labels: List[str]


def normalize_semantic_label(label: str) -> str:
    normalized = label.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return _SEMANTIC_LABEL_ALIASES.get(normalized, normalized)


def metric_to_px(x: float, y: float, n_cells: int, cell_size: float) -> Tuple[int, int]:
    center = n_cells // 2
    epsilon = 1e-9
    return (
        int(x / cell_size + center + epsilon),
        int(y / cell_size + center + epsilon),
    )


def get_semantic_safety_radius_cells(
    label: str,
    query_label: str,
    max_query_radius_cells: Optional[int],
) -> int:
    normalized_label = normalize_semantic_label(label)
    normalized_query = normalize_semantic_label(query_label)
    if normalized_label == normalized_query:
        return 0
    radius = _SEMANTIC_SAFETY_RADIUS_CELLS.get(normalized_label, _DEFAULT_SAFETY_RADIUS_CELLS)
    return max(int(radius), 0)


def _iter_object_metric_points(objects: Sequence, is_gibson: bool) -> Iterable[Tuple[float, float]]:
    for obj in objects:
        if is_gibson:
            points = np.asarray(obj, dtype=np.float32)
            if points.ndim == 1 and points.shape[0] >= 2:
                yield float(points[0]), float(points[1])
                continue
            for pt in points:
                if len(pt) >= 2:
                    yield float(pt[0]), float(pt[1])
            continue

        center = np.asarray(obj.bbox.center, dtype=np.float32)
        if center.shape[0] < 3:
            continue
        yield float(-center[2]), float(-center[0])


def _build_label_seed_map(
    object_locations: Dict[str, Sequence],
    n_cells: int,
    cell_size: float,
    is_gibson: bool,
) -> Tuple[np.ndarray, List[str]]:
    labels: List[str] = []
    label_to_idx: Dict[str, int] = {}
    seed_map = np.zeros((n_cells, n_cells), dtype=np.uint16)

    for raw_label, objects in object_locations.items():
        label = normalize_semantic_label(raw_label)
        if label in _NON_OBSTACLE_LABELS:
            continue
        if label not in label_to_idx:
            label_to_idx[label] = len(labels) + 1
            labels.append(label)
        label_idx = label_to_idx[label]

        for x, y in _iter_object_metric_points(objects, is_gibson):
            px, py = metric_to_px(x, y, n_cells=n_cells, cell_size=cell_size)
            if 0 <= px < n_cells and 0 <= py < n_cells:
                seed_map[px, py] = label_idx

    return seed_map, labels


def _make_disk_kernel(radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    diameter = radius_cells * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))


def build_semantic_collision_data(
    object_locations: Dict[str, Sequence],
    n_cells: int,
    size: float,
    is_gibson: bool,
    query_label: str,
    max_query_radius_cells: int,
) -> SemanticCollisionData:
    cell_size = size / float(n_cells)
    seed_map, labels = _build_label_seed_map(object_locations, n_cells, cell_size, is_gibson)

    collision_map = np.zeros((n_cells, n_cells), dtype=bool)
    label_map = np.zeros((n_cells, n_cells), dtype=np.uint16)

    for idx, label in enumerate(labels, start=1):
        class_seed = (seed_map == idx).astype(np.uint8)
        if not class_seed.any():
            continue

        radius_cells = get_semantic_safety_radius_cells(label, query_label, max_query_radius_cells)
        kernel = _make_disk_kernel(radius_cells)
        dilated = cv2.dilate(class_seed, kernel, iterations=1).astype(bool)
        collision_map |= dilated
        label_map[dilated] = idx

    return SemanticCollisionData(
        collision_map=collision_map,
        label_map=label_map,
        labels=labels,
    )
