"""Motion gap interventions are audits, never a changed V18 forward path."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from real_motion.v18_motion_gap import motion_states, MotionErrors, interaction_decomposition, numpy


def example(n=2):
    center = np.column_stack((np.arange(n), np.zeros(n))).astype(np.float32)
    kta = np.tile(np.arange(1, 7, dtype=np.float32)[None, :, None], (n, 1, 2))
    target = kta * 1.1
    r = {"source_centroid_xy_t0_m": center, "kta_displacement_xy_m": kta,
         "anchors_xy_t0_m": center[:, None] + kta,
         "target_source_displacement_xy_m": target, "target_source_residual_xy_m": target-kta,
         "target_yaw_rad": np.full((n, 6), .2, np.float32),
         "se2_target_valid": np.ones((n, 6), bool), "yaw_label_valid": np.ones((n, 6), bool),
         "source_class_id": np.full(n, 4), "supervised_source": np.arange(n) % 2 == 0,
         "yaw_enabled": np.ones(n, bool), "frame_motion_features": np.ones((n, 6, 5), np.float32)}
    o = {"residual_xy_m": np.full((n, 6, 2), .05, np.float32), "yaw_delta_rad": np.full((n, 6), .1, np.float32)}
    return r, o


class MotionGapTests(unittest.TestCase):
    def test_xy_and_yaw_interventions_only_change_the_named_state(self):
        r, o = example(); s = motion_states(r, o)
        np.testing.assert_array_equal(s["GT_XY_PRED_YAW"][1], s["V18_BASE"][1])
        np.testing.assert_array_equal(s["PRED_XY_GT_YAW"][0], s["V18_BASE"][0])
        np.testing.assert_array_equal(s["GT_XY_GT_YAW"][0], r["anchors_xy_t0_m"]+r["target_source_residual_xy_m"])
        np.testing.assert_array_equal(s["GT_XY_GT_YAW_SUPERVISED_ONLY"][0][1], s["V18_BASE"][0][1])
        np.testing.assert_array_equal(s["GT_XY_GT_YAW_SUPERVISED_ONLY"][1][1], s["V18_BASE"][1][1])

    def test_missing_gt_keeps_v18_instead_of_disappearing_or_zeroing_motion(self):
        r, o = example(); r["se2_target_valid"][0, 2] = False; r["yaw_label_valid"][1, 3] = False
        s = motion_states(r, o)
        for name in ("GT_XY_PRED_YAW", "PRED_XY_GT_YAW", "GT_XY_GT_YAW"):
            for field in (0, 1): np.testing.assert_array_equal(s[name][field][0, 2], s["V18_BASE"][field][0, 2])
        self.assertEqual(s["GT_XY_GT_YAW"][1][1, 3], s["V18_BASE"][1][1, 3])

    def test_pedestrian_class_rule_not_future_yaw_validity_controls_baseline(self):
        r, o = example(); r["source_class_id"][1] = 7; r["yaw_enabled"][1] = False
        s = motion_states(r, o)
        for _, yaw in s.values(): self.assertTrue((yaw[1] == 0).all())
        r["yaw_enabled"][1] = True
        with self.assertRaises(ValueError): motion_states(r, o)

    def test_pivot_and_target_contract_checks(self):
        r, o = example(); bad = copy.deepcopy(r); bad["anchors_xy_t0_m"][0, 2, 0] += .5
        with self.assertRaises(ValueError): motion_states(bad, o)
        bad = copy.deepcopy(r); bad["target_source_displacement_xy_m"][0, 2, 0] += .5
        with self.assertRaises(ValueError): motion_states(bad, o)
        o["yaw_delta_rad"][0, 1] = np.nan
        with self.assertRaises(ValueError): motion_states(r, o)

    def test_no_input_mutation_and_empty_source_support(self):
        r, o = example(); original = copy.deepcopy(r); motion_states(r, o)
        for k in r: np.testing.assert_array_equal(r[k], original[k])
        r, o = example(0); states = motion_states(r, o)
        self.assertEqual(states["V18_BASE"][0].shape, (0, 6, 2))
        errors = MotionErrors(); errors.update(r, o); self.assertEqual(errors.compute()["all"]["source_center_error_m"]["count"], 0)

    def test_continuity_masks_do_not_differentiate_through_missing_gt(self):
        r, o = example(1); r["se2_target_valid"][0] = [True, False, True, True, False, True]
        errors = MotionErrors(); errors.update(r, o); row = errors.compute()["all"]
        self.assertEqual(row["source_center_error_m"]["count"], 4)
        self.assertEqual(row["velocity_error_mps"]["count"], 2)
        self.assertEqual(row["acceleration_error_mps2"]["count"], 0)

    def test_perfect_prediction_zero_errors_and_strata(self):
        r, o = example(); o["residual_xy_m"] = r["target_source_residual_xy_m"]; o["yaw_delta_rad"] = r["target_yaw_rad"]
        errors = MotionErrors(); errors.update(r, o); report = errors.compute()
        self.assertLess(report["all"]["source_center_error_m"]["mean"], 1e-6)
        self.assertEqual(report["all"]["yaw_error_deg"]["mean"], 0.)
        self.assertEqual(report["history_valid/6"]["source_center_error_m"]["count"], 12)
        self.assertEqual(report["supervised/yes"]["source_center_error_m"]["count"], 6)

    def test_interaction_not_independent_additive_effects(self):
        reports = {k: {"metrics": {"mIoU": v}} for k, v in
                   (("V18_BASE", 40.), ("GT_XY_PRED_YAW", 44.), ("PRED_XY_GT_YAW", 41.),
                    ("GT_XY_GT_YAW", 46.), ("GT_XY_GT_YAW_SUPERVISED_ONLY", 45.))}
        d = interaction_decomposition(reports)
        self.assertEqual(d["interaction_pp"], 1.)
        self.assertEqual(d["xy_shapley_diagnostic_pp"], 4.5)
        self.assertEqual(d["yaw_shapley_diagnostic_pp"], 1.5)
        self.assertEqual(d["additional_unsupervised_joint_pp"], 1.)

    def test_tensor_and_bfloat_outputs(self):
        r, o = example(); r = {k: torch.from_numpy(v) for k, v in r.items()}
        o = {k: torch.from_numpy(v).bfloat16() for k, v in o.items()}
        self.assertEqual(numpy(o["residual_xy_m"]).dtype, np.float32)
        self.assertEqual(motion_states(r, o)["V18_BASE"][0].shape, (2, 6, 2))

    def test_full_audit_uses_real_renderer_metrics_and_reference_guard(self):
        from real_motion.geometry import OccupancyGrid
        from real_motion.rigid_transport import RasterizedRigidComponent
        from tools.real_motion import eval_p0_f9_v18_motion_gap as evaluator
        from tools.real_motion.summarize_p0_f9_v18_motion_gap import summarize
        grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(12, 12, 2))
        pcfg = SimpleNamespace(grid=grid, free_label=17, frame_dt_s=.5)
        r, o = example(1); r["source_centroid_xy_t0_m"][:] = 1.5
        r["anchors_xy_t0_m"] = r["source_centroid_xy_t0_m"][:, None]+r["kta_displacement_xy_m"]
        # External source/teacher adapters only are synthetic. A1, raster,
        # delta counting, audit CLI, serialization and reference check are real.
        r.update(scene_name="scene", t0_token="t0", future_tokens=tuple(f"f{i}" for i in range(6)))
        r = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in r.items()}
        comp = {"class_id": 4, "voxel_indices": np.array([[1, 1, 0]]), "centroid_world": np.array([1.5, 1.5, .5])}
        prior = RasterizedRigidComponent(4, comp["voxel_indices"], 1)
        b = np.full(grid.shape_hwd, 17, np.uint8); b[1, 1, 0] = 4; b[11, 11, 0] = 11
        state = {"rec": r, "current": [comp], "current_pose": np.eye(4), "future_poses": [np.eye(4)]*6,
            "source_world_points": [np.array([[1.5, 1.5, .5]])], "source_rel_xy": [np.zeros((1, 2))],
            "world_to_future": [np.eye(4)]*6, "source_z_t0": np.array([.5]), "gpu": {},
            "anchors": [b]*6, "baseline_by_hi": [[prior]]*6,
            "baseline_clear_flat_by_hi": [np.array([26])]*6}
        # Proper flattened index for (1,1,0) in this synthetic grid.
        state["baseline_clear_flat_by_hi"] = [np.array([np.ravel_multi_index((1,1,0), b.shape)])]*6
        outputs = {k: torch.from_numpy(v) for k, v in o.items()}
        truth = b.copy(); truth[1, 1, 0] = 17; truth[5, 5, 0] = 4
        raw = {"future_gt_occ": [truth]*6}
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {k: root/k for k in ("val_cache", "population_manifest", "checkpoint", "info_pkl")}
            for p in paths.values(): p.touch()
            manifest = {"parent_keys": [("scene", str(i)) for i in range(512)], "selected_key_fingerprint": "b"*64}
            argv = ["audit", "--config", str(Path(__file__).resolve().parents[1]/"configs/real_motion_occfm.yaml"),
                    "--dataroot", str(root), "--device", "cpu", "--out-dir", str(root/"result")]
            for name, path in paths.items(): argv += ["--"+name.replace("_", "-"), str(path)]
            with patch.object(evaluator, "make_prepare_config", return_value=pcfg), \
                 patch.object(evaluator, "load_manifest", return_value=(manifest, [("scene", "t0")]*64, None)), \
                 patch.object(evaluator, "load_cache", return_value=(None, [r])), \
                 patch.object(evaluator, "align_records", return_value=[r]), \
                 patch.object(evaluator.full, "_load_model", return_value=({}, model, None)), \
                 patch.object(evaluator, "validate_clean_e14_checkpoint", return_value="a"*64), \
                 patch.object(evaluator, "NuScenesWindowSource", return_value=SimpleNamespace(nusc=None)), \
                 patch.object(evaluator, "window_from_record", return_value=SimpleNamespace(scene_name="scene", t0_token="t0", future_tokens=r["future_tokens"])), \
                 patch.object(evaluator, "load_nuscenes_window_raw", return_value=raw), \
                 patch.object(evaluator.runtime, "_prepare_record", return_value=state), \
                 patch.object(evaluator.runtime, "_stage_gpu_inputs"), patch.object(evaluator.runtime, "_release_gpu_inputs"), \
                 patch.object(evaluator, "assert_forward_exact"), patch.object(evaluator.runtime, "_exactness_check"), \
                 patch.object(evaluator.runtime, "_model_forward", return_value=outputs), \
                 patch.object(evaluator, "gt_moving_support_sequence", return_value=[(np.ones_like(b, bool), None)]*6), \
                 patch("sys.argv", argv):
                evaluator.main()
            data = json.loads((root/"result"/"audit.json").read_text(encoding="utf-8"))
            self.assertFalse(data["training_performed"])
            self.assertEqual(data["variants"]["V18_BASE"]["delta_vs_v18_pp"]["mIoU"], 0.)
            self.assertIn("POSITION / YAW", summarize(data))
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            reference = {"protocol": "p0_f9_source_evidence_audit_v1", "arguments": data["arguments"],
                **{k: data[k] for k in ("checkpoint_sha256", "selected_key_fingerprint", "config_sha256")},
                "variants": {"V18_BASE": data["variants"]["V18_BASE"], "T0_GT_MOTION": data["variants"]["GT_XY_GT_YAW"]}}
            self.assertTrue(evaluator.check_reference(data, reference)["exact_within_1e_8"])
            reference["checkpoint_sha256"] = "c"*64
            with self.assertRaises(RuntimeError): evaluator.check_reference(data, reference)


if __name__ == "__main__": unittest.main()
