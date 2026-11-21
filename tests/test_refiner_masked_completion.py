import unittest

import numpy as np

from real_motion.geometry import OccupancyGrid
from tools.refiner_masked_completion.evaluate import compare, describe, MODELS, REGIONS
from tools.refiner_masked_completion.masking import depth_for, mask_history, regions


class MaskedCompletionTest(unittest.TestCase):
    def setUp(self):
        self.grid = OccupancyGrid(x_min=-4, y_min=-4, z_min=-1,
                                  voxel_size=(1, 1, 1), shape_hwd=(8, 8, 2))

    def test_world_consistent_mask_and_dynamic_input(self):
        labels = np.full((4, 8, 8, 2), 11, np.uint8)
        labels[0, 7, 1, 1] = 4
        labels[-1, 7, 1, 1] = 4
        observed = np.ones_like(labels, bool)
        poses = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
        poses[1, 0, 3] = -2
        poses[2, :2, :2] = [[0, -1], [1, 0]]
        static = np.ones((4, 4, 4, 2), np.float32)
        dynamic_points = object()
        raw = dict(history_occ=labels, history_observed=observed, history_poses=poses)
        history = (dict(memory_reads=dict(static_valid=static, world=dynamic_points)),)
        prior, known, memory = mask_history(raw, history, self.grid, 2)
        self.assertFalse(known[0, 6, 0, 0])
        self.assertTrue(known[0, 7, 1, 1])
        self.assertTrue(known[1].all())
        self.assertFalse(known[2, 0, 0, 0])
        self.assertTrue(known[2, 7, 7, 0])
        self.assertTrue(np.all(prior['history_occ'][-1, 6:] == 17))
        self.assertTrue(np.all(memory[0]['memory_reads']['static_valid'][0, -1] == 0))
        self.assertIs(memory[0]['memory_reads']['world'], dynamic_points)
        self.assertTrue(static.all() and observed.all())
        self.assertEqual(labels[-1, 7, 1, 1], 4)

    def test_natural_region_requires_absence_from_all_history_grids(self):
        past = np.broadcast_to(np.eye(4), (4, 4, 4)).copy()
        past[0, 0, 3] = 1
        future = np.eye(4)
        future[0, 3] = 3
        location = regions(self.grid, past, future, 2)
        np.testing.assert_array_equal(location['natural_front_3s'][:, 4, 0],
                                      [False, False, False, False, False, False, True, True])
        np.testing.assert_array_equal(location['artificial_cut_3s'][:, 4, 0],
                                      [False, False, False, True, True, False, False, False])

    def test_metrics_use_clean_only_for_real_uncovered_region(self):
        gt = np.full((3, 2, 1, 1), 11, np.uint8)
        clean = np.full((6, 2, 1, 1), 17, np.uint8)
        clean[5, 0, 0, 0] = 11
        cut = clean.copy()
        cut[5, :, 0, 0] = [17, 11]
        counts = {name: {key: np.zeros((3 if key == 'full_grid' else 1, 18, 18), np.int64)
                         for key in REGIONS} for name in MODELS}
        footprint = np.array([True, False]).reshape(2, 1, 1)
        compare(counts, dict.fromkeys(MODELS, (clean, cut)), gt,
                dict(natural_front_3s=footprint, artificial_cut_3s=~footprint))
        for model in MODELS:
            scores = describe(counts[model])
            self.assertEqual(scores['natural_front_3s']['classes']['road']['iou'], 100.)
            self.assertEqual(scores['artificial_cut_3s']['classes']['road']['iou'], 100.)
            self.assertEqual(scores['full_grid']['per_horizon']['3.0']['mIoU'], 50.)

    def test_fixed_population_does_not_depend_on_label_content(self):
        record = dict(scene_name='scene-test', t0_token='sample')
        self.assertIn(depth_for(record, 20261011, force=True), (8, 12, 16))
        self.assertEqual(depth_for(record, 20261011),
                         depth_for(dict(record, future_gt_occ='anything'), 20261011))


if __name__ == '__main__':
    unittest.main()
