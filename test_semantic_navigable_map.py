import unittest

import numpy as np

from mapping.semantic_navigable_map import SemanticNavigableMapUpdater, build_semantic_label_config


class SemanticNavigableMapTests(unittest.TestCase):
    def test_unknown_label_uses_default_radius_one(self):
        config = build_semantic_label_config(["unknown custom object"])
        self.assertEqual(config.labels, ["unknown custom object"])
        self.assertEqual(int(config.radii[0]), 1)

    def test_incremental_update_inflates_only_local_region(self):
        config = build_semantic_label_config(["chair"])
        updater = SemanticNavigableMapUpdater(n_cells=31, label_config=config)

        base = np.ones((31, 31), dtype=bool)
        updater.reset(base)

        changed = np.zeros((31, 31), dtype=bool)
        changed[15, 15] = True

        label_ids = updater.seed_label_map.copy()
        label_ids[15, 15] = 1
        semantic_map = updater.update(base, changed, label_ids)

        blocked = (~semantic_map).astype(np.uint8)
        ys, xs = np.nonzero(blocked)
        self.assertTrue(len(xs) > 0)
        # Chair radius in semantic_collision is 2; local bounds should stay compact.
        self.assertLessEqual(int(xs.max() - xs.min()), 6)
        self.assertLessEqual(int(ys.max() - ys.min()), 6)

    def test_changed_mask_without_label_reverts_to_base_map(self):
        config = build_semantic_label_config(["chair"])
        updater = SemanticNavigableMapUpdater(n_cells=21, label_config=config)

        base = np.ones((21, 21), dtype=bool)
        updater.reset(base)

        changed = np.zeros((21, 21), dtype=bool)
        changed[10, 10] = True

        label_ids = updater.seed_label_map.copy()
        label_ids[10, 10] = 1
        first = updater.update(base, changed, label_ids)
        self.assertFalse(bool(first[10, 10]))

        cleared_ids = updater.seed_label_map.copy()
        cleared_ids[10, 10] = 0
        second = updater.update(base, changed, cleared_ids)
        self.assertTrue(bool(second[10, 10]))


if __name__ == "__main__":
    unittest.main()
