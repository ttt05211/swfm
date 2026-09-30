"""Numpy-only tests; also runnable through stdlib unittest on Windows."""
import unittest
import importlib.util
from pathlib import Path
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

# Load the numpy-only core without real_motion.__init__ (which requires Torch).
# This tests the exact repository source on CPU-only development machines too.
_spec = importlib.util.spec_from_file_location(
    "_source_evidence_audit_core",
    Path(__file__).resolve().parents[1] / "real_motion/source_evidence_audit.py",
)
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)
associate_backwards = _core.associate_backwards
choose_candidate_gt_assisted = _core.choose_candidate_gt_assisted
edit_quality = _core.edit_quality
metric_count_delta = _core.metric_count_delta
planar_move = _core.planar_move
protected_add_indices = _core.protected_add_indices
raster_flat = _core.raster_flat
register_history_shape = _core.register_history_shape
route_diagnostic = _core.route_diagnostic


def counts(p, g, m):
    c = np.bincount((g * 18 + p).reshape(-1), minlength=324).reshape(18, 18)
    mc = np.bincount((g * 18 + p)[m], minlength=324).reshape(18, 18)
    ids = np.array([4, 7])
    return (int(c[:17, :17].sum()), int(c.sum() - c[17, 17]), np.diag(c)[:17],
            (c.sum(0) + c.sum(1) - np.diag(c))[:17], np.diag(mc)[ids],
            (mc.sum(0) + mc.sum(1) - np.diag(mc))[ids])


def component(x, cid=4):
    return {"class_id": cid, "centroid_world": np.array([x, 0.0, 0.0])}


class SourceEvidenceTests(unittest.TestCase):
    def test_planar_motion_preserves_z_and_source_pivot(self):
        pts = np.array([[2., 0., 3.], [1., 1., 4.]])
        result = planar_move(pts, [1, 0, 0], [5, 6, 9], np.pi / 2)
        np.testing.assert_allclose(result, [[5, 7, 3], [4, 6, 4]], atol=1e-12)

    def test_raster_axis_order_dedup_and_oob(self):
        pts = [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5], [-0.01, 0, 0], [4, 0, 0]]
        ids, oob = raster_flat(pts, np.eye(4), [0, 0, 0], [1, 1, 1], (4, 3, 3))
        np.testing.assert_array_equal(ids, [5])
        self.assertEqual(oob, 2)

    def test_backwards_dictionary_velocity_and_missing(self):
        frames = [[component(0)], [], [component(2)]]
        links, audit = associate_backwards(frames, frames[-1], {0: np.array([2., 0, 0])})
        self.assertEqual(links, [[0, None, 0]])
        self.assertEqual(audit["matched"], 1)

    def test_missing_velocity_is_zero_not_source_index(self):
        frames = [[component(0)], [component(0)]]
        links, _ = associate_backwards(frames, frames[-1], {})
        self.assertEqual(links, [[0, 0]])

    def test_ambiguous_match_fails_closed(self):
        frames = [[component(-0.1), component(0.1)], [component(0)]]
        links, audit = associate_backwards(frames, frames[-1], {})
        self.assertIsNone(links[0][0])
        self.assertEqual(audit["ambiguous"], 1)

    def test_one_to_one_and_class_guard(self):
        frames = [[component(0), component(0, 7)], [component(0), component(0.05)]]
        links, _ = associate_backwards(frames, frames[-1], {})
        self.assertEqual(sum(row[0] is not None for row in links), 1)

    def test_empty_source_population(self):
        links, _ = associate_backwards([[], []], [], {})
        self.assertEqual(links, [])

    def test_registration_translation_and_z_preservation(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(-1, 1, (80, 3))
        q = p + [4, 2, 7]
        r = register_history_shape(p, q)
        self.assertTrue(r.accepted)
        # Registration deliberately works in XY, so a Z offset is not fitted.
        np.testing.assert_allclose(r.points[:, :2], q[np.lexsort(p.T[::-1]), :2], atol=1e-9)
        np.testing.assert_allclose(r.points[:, 2], p[np.lexsort(p.T[::-1]), 2])

    def test_small_registration_rejected(self):
        r = register_history_shape(np.zeros((5, 3)), np.zeros((7, 3)))
        self.assertFalse(r.accepted)

    def test_registration_order_invariance(self):
        p = np.random.default_rng(1).normal(size=(30, 3))
        a = register_history_shape(p, p + [2, 0, 0])
        b = register_history_shape(p[::-1], (p + [2, 0, 0])[::-1])
        np.testing.assert_array_equal(a.points, b.points)

    def test_protected_add_and_identity(self):
        base = np.array([17, 4, 17])
        np.testing.assert_array_equal(protected_add_indices(base, [0, 1], 7), [7, 4, 17])
        np.testing.assert_array_equal(protected_add_indices(base, [], 7), base)
        with self.assertRaises(ValueError):
            protected_add_indices(base, [-1], 7)

    def test_selection_retains_wrong_voxels_in_whole_candidate(self):
        b, g = np.full(5, 17), np.array([4, 4, 17, 17, 17])
        ids, score = choose_candidate_gt_assisted(b, g, [[0, 1, 2]], 4)
        np.testing.assert_array_equal(ids, [0, 1, 2])
        self.assertEqual(score, 1)

    def test_selection_can_abstain_and_has_stable_ties(self):
        b, g = np.full(3, 17), np.array([4, 17, 4])
        ids, score = choose_candidate_gt_assisted(b, g, [[0, 1]], 4)
        self.assertEqual(len(ids), 0)
        ids, score = choose_candidate_gt_assisted(b, g, [[2], [0]], 4)
        np.testing.assert_array_equal(ids, [2])

    def test_metric_updates_add_delete_relabel_exactly(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            b, p, g = (rng.integers(0, 18, (4, 5, 3)) for _ in range(3))
            m = rng.random(b.shape) > 0.5
            got = metric_count_delta(counts(b, g, m), b, p, g, m, [4, 7])
            for actual, expected in zip(got, counts(p, g, m)):
                np.testing.assert_array_equal(actual, expected)

    def test_metric_rejects_bad_labels_and_shapes(self):
        b = np.array([17]); g = np.array([4]); m = np.array([True])
        with self.assertRaises(ValueError):
            metric_count_delta(counts(b, g, m), b, np.array([18]), g, m, [4, 7])
        with self.assertRaises(ValueError):
            metric_count_delta(counts(b, g, m), b, np.array([4, 4]), g, m, [4, 7])

    def test_edit_quality_reports_harm_and_removal(self):
        q = edit_quality(np.array([17, 4, 4]), np.array([4, 17, 7]), np.array([17, 17, 4]))
        self.assertEqual(q["added_occ_tp"], 0)
        self.assertEqual(q["removed_false_occupancy"], 1)
        self.assertEqual(q["damaged"], 2)

    def test_route_never_uses_filtered_voxel_gt_gain(self):
        g = {"T0_GT_MOTION": 2., "HISTORY_GT_ALIGN_GT_MOTION": 2.1,
             "HISTORY_CAUSAL_ALIGN_PRED_MOTION": -0.5, "STATIC_VOXEL_GT_FILTER": 100.}
        self.assertEqual(route_diagnostic(g), "prioritize_observed_source_motion")

    def test_physical_speed_gate_cannot_be_bypassed_by_velocity_prior(self):
        links, audit = associate_backwards([[component(-20)], [component(0)]],
                                          [component(0)], {0: np.array([40., 0., 0.])})
        self.assertIsNone(links[0][0])
        self.assertEqual(audit["unmatched"], 1)

    def test_invalid_registration_and_candidate_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            register_history_shape(np.zeros((8, 3)), np.zeros((8, 3)), max_points=3)
        with self.assertRaises(ValueError):
            register_history_shape(np.full((8, 3), np.nan), np.zeros((8, 3)))
        with self.assertRaises(ValueError):
            choose_candidate_gt_assisted(np.full(3, 17), np.full(3, 4), [[-1]], 4)
        with self.assertRaises(ValueError):
            raster_flat([[np.nan, 0, 0]], np.eye(4), [0, 0, 0], [1, 1, 1], (3, 3, 3))

    def test_inplace_add_is_explicit_and_protects_base(self):
        base = np.array([17, 4, 17])
        result = protected_add_indices(base, [0, 1], 7, copy=False)
        self.assertIs(result, base)
        np.testing.assert_array_equal(base, [7, 4, 17])


@unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                     "Torch-dependent evaluator tests require the server/project environment")
class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.real_motion import eval_p0_f9_source_evidence_audit as audit
        cls.audit = audit

    def test_bfloat16_outputs_convert_without_numpy_dtype_error(self):
        import torch
        np.testing.assert_array_equal(self.audit._numpy(torch.tensor([1., 2.], dtype=torch.bfloat16)), [1., 2.])

    def test_static_selection_keeps_whole_patch_including_wrong_voxels(self):
        b = np.full((4, 4, 2), 17, np.uint8)
        memory, gt = b.copy(), b.copy()
        for xyz in ((0, 0, 0), (0, 1, 1), (1, 1, 0)):
            memory[xyz] = 11
        gt[0, 0, 0] = gt[0, 1, 1] = 11
        result = self.audit._static_predictions(b, memory, gt, 2)
        self.assertEqual(result["STATIC_PATCH_GT_SELECT"][1, 1, 0], 11)
        self.assertEqual(result["STATIC_VOXEL_GT_FILTER"][1, 1, 0], 17)
        np.testing.assert_array_equal(b, np.full_like(b, 17))

    def test_static_abstains_nonpositive_patch_and_protects_occupied(self):
        b = np.full((4, 4, 2), 17, np.uint8)
        b[0, 0, 0] = 4
        memory = np.full_like(b, 11)
        gt = np.full_like(b, 17)
        result = self.audit._static_predictions(b, memory, gt, 2)
        np.testing.assert_array_equal(result["STATIC_PATCH_GT_SELECT"], b)
        self.assertEqual(result["STATIC_MEMORY"][0, 0, 0], 4)

    def test_render_current_uses_full_a1_clear_write_not_gt_filter(self):
        from real_motion.geometry import OccupancyGrid
        from real_motion.rigid_transport import RasterizedRigidComponent
        grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(4, 4, 2))
        b = np.full(grid.shape_hwd, 17, np.uint8)
        b[1, 1, 0], b[3, 3, 0] = 4, 11
        comp = {"class_id": 4, "voxel_indices": np.array([[1, 1, 0]]),
                "centroid_world": np.array([1.5, 1.5, .5])}
        prior = RasterizedRigidComponent(4, comp["voxel_indices"], 1)
        state = {"current": [comp], "source_world_points": [np.array([[1.5, 1.5, .5]])],
                 "source_rel_xy": [np.zeros((1, 2))], "world_to_future": [np.eye(4)] * 6,
                 "anchors": [b] * 6, "baseline_by_hi": [[prior]] * 6,
                 "baseline_clear_flat_by_hi": [np.array([10])] * 6}
        centers = {h: [np.array([2.5, 1.5, .5])] for h in self.audit.REPORT}
        yaw = {h: [0.] for h in self.audit.REPORT}
        predictions, _ = self.audit._render_current(state, SimpleNamespace(grid=grid, free_label=17), centers, yaw)
        for p in predictions:
            self.assertEqual(p[1, 1, 0], 17)
            self.assertEqual(p[2, 1, 0], 4)
            self.assertEqual(p[3, 3, 0], 11)
        self.assertEqual(b[1, 1, 0], 4)

    def test_history_variants_and_common_subset_use_protected_add(self):
        from real_motion.geometry import OccupancyGrid
        grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(4, 4, 2))
        b = np.full(grid.shape_hwd, 17, np.uint8)
        b[0, 0, 0] = 11
        gt = np.full_like(b, 17)
        gt[1, 1, 0] = gt[2, 1, 0] = 4
        points = np.array([[.5, .5, .5], [1.5, 1.5, .5], [2.5, 1.5, .5], [3.5, 1.5, .5]])
        families = {name: [[points]] for name in ("CAUSAL_ALIGN", "GT_ALIGN", "COMMON_GT_ALIGN", "COMMON_CAUSAL_ALIGN")}
        state = {"current": [component(0)], "world_to_future_current": np.eye(4)}
        rows = self.audit._history_predictions({"V18_BASE": b, "T0_GT_MOTION": b}, families, state, grid,
                         [[0, 0, 0]], [0.], [[0, 0, 0]], [0.], gt, defaultdict(int))
        self.assertEqual(rows["HISTORY_GT_SELECT_PRED_MOTION"][3, 1, 0], 4)
        self.assertEqual(rows["HISTORY_CAUSAL_VOXEL_GT_FILTER"][3, 1, 0], 17)
        for row in rows.values():
            self.assertEqual(row[0, 0, 0], 11)
        np.testing.assert_array_equal(rows["HISTORY_COMMON_GT_ALIGN_GT_MOTION"], rows["HISTORY_COMMON_CAUSAL_ALIGN_GT_MOTION"])
        self.assertEqual(b[1, 1, 0], 17)

    def test_nonfinite_json_is_null_without_metric_changes(self):
        self.assertEqual(self.audit.finite_json({"v": [float("nan"), float("inf"), 1.]}), {"v": [None, None, 1.]})


if __name__ == "__main__":
    unittest.main()
