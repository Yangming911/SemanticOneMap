import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np

from eval.semantic_collision import (
    build_semantic_collision_data,
    get_semantic_safety_radius_cells,
    metric_to_px,
    normalize_semantic_label,
)
from read_timing import load_states


def make_hm3d_object(center_xyz):
    return SimpleNamespace(bbox=SimpleNamespace(center=np.asarray(center_xyz, dtype=np.float32)))


class SemanticCollisionTests(unittest.TestCase):
    def test_normalize_semantic_label_applies_aliases(self):
        self.assertEqual(normalize_semantic_label("TV_Monitor"), "tv")
        self.assertEqual(normalize_semantic_label("potted-plant"), "potted plant")
        self.assertEqual(normalize_semantic_label(" DiningTable "), "dining table")

    def test_query_radius_is_capped_by_detection_distance(self):
        self.assertEqual(get_semantic_safety_radius_cells("chair", "chair", 1), 1)
        self.assertEqual(get_semantic_safety_radius_cells("chair", "tv", 1), 2)

    def test_metric_to_px_maps_origin_to_grid_center(self):
        self.assertEqual(metric_to_px(0.0, 0.0, n_cells=11, cell_size=1.0), (5, 5))

    def test_build_semantic_collision_data_for_hm3d_objects(self):
        data = build_semantic_collision_data(
            object_locations={"chair": [make_hm3d_object([0.0, 0.0, 0.0])]},
            n_cells=11,
            size=11.0,
            is_gibson=False,
            query_label="tv",
            max_query_radius_cells=1,
        )

        center = metric_to_px(0.0, 0.0, n_cells=11, cell_size=1.0)
        self.assertEqual(data.labels, ["chair"])
        self.assertTrue(data.collision_map[center])
        self.assertEqual(data.label_map[center], 1)
        self.assertGreater(int(data.collision_map.sum()), 1)

    def test_query_object_uses_capped_radius_in_collision_map(self):
        capped = build_semantic_collision_data(
            object_locations={"chair": [make_hm3d_object([0.0, 0.0, 0.0])]},
            n_cells=21,
            size=21.0,
            is_gibson=False,
            query_label="chair",
            max_query_radius_cells=1,
        )
        uncapped = build_semantic_collision_data(
            object_locations={"chair": [make_hm3d_object([0.0, 0.0, 0.0])]},
            n_cells=21,
            size=21.0,
            is_gibson=False,
            query_label="tv",
            max_query_radius_cells=1,
        )

        self.assertLess(int(capped.collision_map.sum()), int(uncapped.collision_map.sum()))

    def test_build_semantic_collision_data_for_gibson_points(self):
        gibson_points = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        data = build_semantic_collision_data(
            object_locations={"tv_monitor": [gibson_points]},
            n_cells=15,
            size=15.0,
            is_gibson=True,
            query_label="chair",
            max_query_radius_cells=1,
        )

        first_px = metric_to_px(0.0, 0.0, n_cells=15, cell_size=1.0)
        second_px = metric_to_px(2.0, 0.0, n_cells=15, cell_size=1.0)
        self.assertEqual(data.labels, ["tv"])
        self.assertEqual(data.label_map[first_px], 1)
        self.assertEqual(data.label_map[second_px], 1)
        self.assertTrue(data.collision_map[first_px])
        self.assertTrue(data.collision_map[second_px])

    def test_read_timing_load_states_includes_semantic_collision(self):
        with TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/state_3.txt", "w") as f:
                f.write("7")

            states, result_names = load_states(tmpdir)

        self.assertEqual(states, {3: 7})
        self.assertEqual(result_names[7], "SEMANTIC_COLLISION")


if __name__ == "__main__":
    unittest.main()
