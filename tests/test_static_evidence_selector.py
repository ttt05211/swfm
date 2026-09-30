"""Synthetic contracts and actual CPU train/serialize/infer tests (no nuScenes)."""
from __future__ import annotations
import copy
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS
from real_motion.static_evidence_selector import (
    PROTOCOL, FEATURE_PROTOCOL, THRESHOLD, HISTORY_DIM, CONTEXT_DIM,
    patch_layout, patch_descriptors, assemble_inputs, prepare_selector_inputs,
    compose_selected_static, supervision_counts, sample_training_patches,
    sparse_static_counts, acceptance_gate,
)
from real_motion.static_evidence_selector_model import StaticEvidenceSelector, SelectorConfig, utility_loss
from real_motion.v22_causal_emergence import build_future_static_memory_only


def scene():
    grid = OccupancyGrid(x_min=0, y_min=0, z_min=0, voxel_size=(1, 1, 1), shape_hwd=(8, 8, 4))
    b = np.full(grid.shape_hwd, 17, np.uint8)
    b[0, 0, 0] = 4
    memory = np.full_like(b, 17)
    memory[:4, :4, 1] = 11
    memory[4:, 4:, 2] = 13
    hist = np.zeros((6, 4, HISTORY_DIM), np.float32)
    return grid, b, memory, assemble_inputs(hist, b, memory, 1.)


class CoreTests(unittest.TestCase):
    def test_uneven_grid_capacities_and_axis_order(self):
        p, cap, hw = patch_layout((5, 7, 3))
        self.assertEqual(hw, (2, 2)); self.assertEqual(int(cap.sum()), 105)
        self.assertEqual(p[np.ravel_multi_index((4, 6, 2), (5, 7, 3))], 3)
        np.testing.assert_array_equal(cap, [48, 36, 12, 9])

    def test_descriptor_height_coverage_and_missing_observation(self):
        sem = np.full((4, 4, 4), 17, np.uint8)
        known = np.zeros_like(sem, bool)
        sem[0, 0, 2], known[0, 0, 2] = 11, True
        d = patch_descriptors(sem, known)[0]
        self.assertEqual(d[11], 1/64); self.assertEqual(d[18], 1/64)
        self.assertEqual(d[19], 2.5/4); self.assertEqual(d[20], 0.)

    def test_no_gt_feature_argument_or_gt_dependent_features(self):
        _, b, memory, x = scene()
        self.assertNotIn("future_gt", inspect.signature(prepare_selector_inputs).parameters)
        y = assemble_inputs(np.zeros((6, 4, HISTORY_DIM)), b, memory, 1.)
        gt_a, gt_b = memory, np.full_like(memory, 17)
        a, _ = supervision_counts(x, gt_a); c, _ = supervision_counts(x, gt_b)
        self.assertGreater(a.sum(), c.sum())
        np.testing.assert_array_equal(x.history, y.history)
        np.testing.assert_array_equal(x.context, y.context)

    def test_protected_whole_patch_all_reject_identity_all_keep_exact(self):
        _, b, memory, x = scene()
        reject = compose_selected_static(b, x, np.zeros(len(x.patch_ids)))
        np.testing.assert_array_equal(reject, b)
        keep = compose_selected_static(b, x, np.ones(len(x.patch_ids)))
        expected = b.copy(); eligible = (b == 17) & (memory != 17); expected[eligible] = memory[eligible]
        np.testing.assert_array_equal(keep, expected)
        take_one = compose_selected_static(b, x, np.array([1., 0.]))
        self.assertEqual((take_one[:4, :4, 1] == 11).sum(), 16)
        self.assertEqual(take_one[0, 0, 0], 4)

    def test_no_dynamic_proposals_and_no_overwrite(self):
        _, b, m, x = scene()
        m[3, 3, 2] = 7
        with self.assertRaises(ValueError): assemble_inputs(np.zeros((6, 4, HISTORY_DIM)), b, m, 1.)
        modified = b.copy(); modified.reshape(-1)[x.voxel_flat[0]] = 4
        with self.assertRaises(ValueError): compose_selected_static(modified, x, np.ones(len(x.patch_ids)))
        with self.assertRaises(ValueError): compose_selected_static(b, x, np.full(len(x.patch_ids), np.nan))

    def test_exact_memory_parity_random_poses_and_latest_clearing(self):
        grid, b, _, _ = scene()
        rng = np.random.default_rng(43)
        history = rng.integers(0, 18, size=(6, *grid.shape_hwd), dtype=np.uint8)
        observed = rng.random(history.shape) > .25
        poses = [np.eye(4) for _ in range(6)]
        for i, p in enumerate(poses): p[:2, 3] = [.23 * i, -.15 * i]
        future = [np.eye(4) for _ in range(6)]
        for i, p in enumerate(future): p[0, 3] = .31 * (i + 6)
        reference = build_future_static_memory_only(history, observed, poses, future, grid=grid,
            dynamic_class_ids=DYNAMIC_CLASS_IDS, workers=1)
        inputs, memories = prepare_selector_inputs(history, observed, poses, future, [b] * 6, grid=grid, workers=3)
        np.testing.assert_array_equal(np.stack(memories), reference)
        for x in inputs:
            self.assertEqual(tuple(x.history.shape[1:]), (6, 5, HISTORY_DIM))
            self.assertEqual(x.context.shape[1], CONTEXT_DIM)
        history.fill(17); observed.fill(True); history[0, 2, 2, 1] = 11
        _, cleared = prepare_selector_inputs(history, observed, [np.eye(4)]*6, [np.eye(4)]*6, [b]*6, grid=grid)
        self.assertTrue((np.stack(cleared) == 17).all())

    def test_balanced_sampling_preserves_population_weights(self):
        good = np.array([8., 6., 1., 0., 1., 0.])
        bad = np.array([0., 1., 6., 8., 9., 5.])
        ids, w = sample_training_patches(good, bad, 4, np.random.default_rng(0))
        self.assertEqual((good[ids] > bad[ids]).sum(), 2)
        self.assertAlmostEqual(float(w[good[ids] > bad[ids]].sum()), 2.)
        self.assertAlmostEqual(float(w[good[ids] <= bad[ids]].sum()), 4.)
        all_ids, all_w = sample_training_patches(good, bad, 99, np.random.default_rng(0))
        self.assertEqual(len(np.unique(all_ids)), 6); np.testing.assert_array_equal(all_w, np.ones(6))
        with self.assertRaises(ValueError): sample_training_patches(good, bad, 1, np.random.default_rng(0))
        ids, w = sample_training_patches(np.zeros(3), np.ones(3), 2, np.random.default_rng(0))
        self.assertEqual(len(ids), 2); self.assertAlmostEqual(float(w.sum()), 3.)

    def test_neighbor_padding_does_not_wrap_across_grid(self):
        _, b, m, _ = scene()
        hist = np.arange(6 * 4 * HISTORY_DIM, dtype=np.float32).reshape(6, 4, HISTORY_DIM)
        x = assemble_inputs(hist, b, m, 1.)
        np.testing.assert_array_equal(x.history[0, :, 1], np.zeros((6, HISTORY_DIM)))
        np.testing.assert_array_equal(x.history[0, :, 3], np.zeros((6, HISTORY_DIM)))
        np.testing.assert_array_equal(x.history[0, :, 4], hist[:, 1].astype(np.float16))

    def test_empty_proposals_and_budget_gate(self):
        _, b, _, _ = scene()
        x = assemble_inputs(np.zeros((6, 4, HISTORY_DIM)), b, np.full_like(b, 17), 1.)
        self.assertEqual(x.history.shape, (0, 6, 5, HISTORY_DIM))
        np.testing.assert_array_equal(compose_selected_static(b, x, np.empty(0)), b)
        d = {"mIoU": .5, "MovingMacro": 0., "MovingMicro": 0.,
             "per_horizon": {str(h): {"mIoU": .1, "MovingMacro": 0., "MovingMicro": 0.} for h in (1., 2., 3.)}}
        self.assertTrue(acceptance_gate(d)["pass"])
        d["per_horizon"]["1.0"]["mIoU"] = -.01
        self.assertFalse(acceptance_gate(d)["pass"])


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        from tools.real_motion import static_evidence_selector_common as common
        from tools.real_motion.train_p0_f9_static_evidence_selector import training_step
        cls.common, cls.training_step = common, staticmethod(training_step)

    def test_initialization_is_reject_and_has_finite_gradient(self):
        _, _, _, x = scene()
        model = StaticEvidenceSelector()
        logits = model(torch.from_numpy(x.history), torch.from_numpy(x.context))
        self.assertTrue((torch.sigmoid(logits) < .5).all())
        loss = utility_loss(logits, torch.ones_like(logits), torch.ones_like(logits), torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(model.keep_head.bias.grad).all())
        self.assertIsNone(getattr(model, "semantic_head", None))

    def test_utility_bce_calibrated_optimum_and_invalid_objective(self):
        p = torch.tensor([.8], requires_grad=True)
        loss = utility_loss(torch.logit(p), torch.tensor([8.]), torch.tensor([2.]), torch.ones(1))
        loss.backward(); self.assertAlmostEqual(float(p.grad), 0., places=5)
        with self.assertRaises(ValueError): utility_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.ones(1))

    def test_sparse_metrics_equal_dense_including_dynamic_gt_and_fp(self):
        _, b, _, x = scene()
        rng = np.random.default_rng(44)
        gt = rng.integers(0, 18, size=b.shape, dtype=np.uint8)
        moving = rng.random(b.shape) > .5
        from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
        before = Metrics.counts(b, gt, moving, 17)
        p = np.array([1., 0.])
        after = compose_selected_static(b, x, p)
        exact = Metrics.counts(after, gt, moving, 17)
        got = sparse_static_counts(before, x.labels, gt.reshape(-1)[x.voxel_flat], p[x.voxel_patch_rows] >= .5)
        for a, c in zip(exact, got): np.testing.assert_array_equal(a, c)
        for a, c in zip(got[4:], before[4:]): np.testing.assert_array_equal(a, c)

    def test_actual_training_learns_synthetic_utility(self):
        torch.manual_seed(42)
        model = StaticEvidenceSelector(SelectorConfig(width=32, layers=1))
        rng = np.random.default_rng(42)
        n = 32
        history = rng.normal(size=(n, 6, 5, HISTORY_DIM)).astype(np.float16) * .01
        context = np.zeros((n, CONTEXT_DIM), np.float16)
        context[:16, 11] = 1.; context[16:, 13] = 1.
        bank = {"history": history, "context": context, "correct": np.r_[np.full(16, 8.), np.zeros(16)].astype(np.float32),
                "wrong": np.r_[np.zeros(16), np.full(16, 8.)].astype(np.float32), "weight": np.ones(n, np.float32)}
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        losses = [self.training_step(model, opt, bank, np.arange(n), torch.device("cpu"))["loss"] for _ in range(50)]
        model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(torch.from_numpy(history), torch.from_numpy(context))).numpy()
        self.assertLess(losses[-1], losses[0] * .2)
        self.assertTrue((p[:16] >= .5).all()); self.assertTrue((p[16:] < .5).all())
        # Actual disk serialization through the formal checkpoint loader.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.pt"
            value = {**model.contract(), "threshold": THRESHOLD, "state_dict": model.state_dict(),
                     "base_checkpoint_sha256": "a"*64, "runtime_config_fingerprint": "b"*64}
            self.common.atomic_checkpoint(path, value)
            _, loaded = self.common.load_selector(path, torch.device("cpu"), base_sha="a"*64, config_sha="b"*64)
            loaded.eval()
            with torch.no_grad(): restored = torch.sigmoid(loaded(torch.from_numpy(history), torch.from_numpy(context))).numpy()
            np.testing.assert_array_equal(restored, p)
            with self.assertRaises(RuntimeError): self.common.load_selector(path, torch.device("cpu"), base_sha="c"*64)

    def test_deployment_requests_no_future_gt(self):
        _, b, _, x = scene()
        model = StaticEvidenceSelector()
        calls = []
        class Provider:
            device = torch.device("cpu")
            def prepare(self, source, record, *, include_gt):
                calls.append(include_gt)
                return "window", {"future_gt_occ": None}, [b]*6, [x]*6, []
        _, predictions = self.common.forecast_with_selector(Provider(), None, {}, model)
        self.assertEqual(calls, [False])
        for p in predictions: np.testing.assert_array_equal(p, b)

    def test_resume_matches_uninterrupted_optimizer_rng_and_weights(self):
        from tools.real_motion.train_p0_f9_static_evidence_selector import restore_training_checkpoint
        torch.manual_seed(45)
        model = StaticEvidenceSelector(SelectorConfig(width=32, layers=1))
        rng = np.random.default_rng(45)
        n = 8
        bank = {"history": rng.normal(size=(n, 6, 5, HISTORY_DIM)).astype(np.float16),
                "context": rng.normal(size=(n, CONTEXT_DIM)).astype(np.float16),
                "correct": np.arange(n, dtype=np.float32), "wrong": np.full(n, 8., np.float32),
                "weight": np.ones(n, np.float32)}
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        self.training_step(model, opt, bank, rng.integers(n, size=4), torch.device("cpu"))
        identity = {"base_checkpoint_sha256": "a"*64, "runtime_config_fingerprint": "b"*64,
                    "bank_fingerprint": self.common.bank_fingerprint(bank)}
        ck = {**model.contract(), **identity, "threshold": THRESHOLD, "mode": "screen",
              "checkpoint_role": "resume_last", "state_dict": copy.deepcopy(model.state_dict()),
              "optimizer": copy.deepcopy(opt.state_dict()), "sampling_rng_state": copy.deepcopy(rng.bit_generator.state),
              "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": [], "successful_updates": 1}
        next_ids = rng.integers(n, size=4)
        next_stats = self.training_step(model, opt, bank, next_ids, torch.device("cpu"))
        next_torch = torch.rand(5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            self.common.atomic_checkpoint(path, ck)
            _, restored, optimizer, restored_rng = restore_training_checkpoint(path, torch.device("cpu"), identity, "screen")
            restored_ids = restored_rng.integers(n, size=4)
            np.testing.assert_array_equal(next_ids, restored_ids)
            stats = self.training_step(restored, optimizer, bank, restored_ids, torch.device("cpu"))
            self.assertEqual(next_stats, stats)
            torch.testing.assert_close(torch.rand(5), next_torch, rtol=0, atol=0)
            for a, b in zip(model.parameters(), restored.parameters()): torch.testing.assert_close(a, b, rtol=0, atol=0)
            changed = dict(identity, bank_fingerprint="c"*64)
            with self.assertRaises(RuntimeError): restore_training_checkpoint(path, torch.device("cpu"), changed, "screen")
            ck["checkpoint_role"] = "inference_best"
            self.common.atomic_checkpoint(path, ck)
            with self.assertRaises(RuntimeError): restore_training_checkpoint(path, torch.device("cpu"), identity, "screen")

    def test_preparation_samples_once_per_window_without_dense_disk_cache(self):
        from tools.real_motion.train_p0_f9_static_evidence_selector import prepare_bank
        _, b, memory, inputs = scene()
        calls = []
        class Provider:
            def prepare(self, source, record, *, include_gt):
                calls.append((record, include_gt))
                return None, {"future_gt_occ": [memory]*6}, [b]*6, [inputs]*6, [memory]*6
        bank = prepare_bank(Provider(), None, ["a", "b"], seed=45)
        self.assertEqual(calls, [("a", True), ("b", True)])
        self.assertEqual(bank["history"].shape, (24, 6, 5, HISTORY_DIM))
        self.assertTrue((bank["wrong"] == 0).all())
        self.assertEqual(set(bank), {"history", "context", "correct", "wrong", "weight"})
        with self.assertRaises(RuntimeError): prepare_bank(Provider(), None, ["a"], seed=45, max_mib=.001)

    def test_model_contract_bad_shape_and_empty(self):
        model = StaticEvidenceSelector()
        self.assertEqual(model(torch.empty(0, 6, 5, HISTORY_DIM), torch.empty(0, CONTEXT_DIM)).shape, (0,))
        with self.assertRaises(ValueError): model(torch.empty(2, 5, 5, HISTORY_DIM), torch.empty(2, CONTEXT_DIM))

    def test_complete_entrypoints_train_resume_summarize_and_gt_free_export(self):
        # Replace ONLY external dataset/teacher adapters. The CLI, training,
        # metrics, checkpoint selection, serialization and export run for real.
        from tools.real_motion import train_p0_f9_static_evidence_selector as train
        from tools.real_motion import eval_p0_f9_static_evidence_selector as evaluation
        from tools.real_motion.summarize_p0_f9_static_evidence_selector import summarize
        grid, b, memory, x = scene()
        gt = memory.copy(); gt[0, 0, 0] = 4
        requests = []
        class Provider:
            def __init__(self, *args):
                self.device = torch.device("cpu"); self.workers = 1
                self.pcfg = SimpleNamespace(grid=grid); self.sha = self_common.CLEAN_SHA256
            def prepare(self, source, record, *, include_gt):
                requests.append(include_gt)
                window = SimpleNamespace(scene_name=record["scene_name"], t0_token=record["t0_token"],
                                         future_tokens=tuple(f"f{i}" for i in range(6)))
                raw = {"future_gt_occ": [gt]*6 if include_gt else None}
                return window, raw, [b]*6, [x]*6, [memory]*6
        self_common = self.common
        train_records = [{"scene_name": "train", "t0_token": f"t{i}"} for i in range(4)]
        dev_records = [{"scene_name": "dev", "t0_token": f"d{i}"} for i in range(64)]
        keys = tuple((r["scene_name"], r["t0_token"]) for r in dev_records)
        manifest = {"parent_keys": [("dev", f"p{i}") for i in range(512)],
                    "manifest_fingerprint": "c"*64, "parent_key_fingerprint": "d"*64,
                    "selected_key_fingerprint": "e"*64}
        moving = [(np.ones(grid.shape_hwd, bool), None)] * 6
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {name: root / name for name in ("train_cache", "dev_cache", "population_manifest", "base_checkpoint", "train_info", "dev_info")}
            for p in paths.values(): p.touch()
            config = Path(__file__).resolve().parents[1] / "configs" / "real_motion_occfm.yaml"
            argv = ["train", "--config", str(config), "--dataroot", str(root), "--device", "cpu",
                    "--mode", "smoke", "--train-windows", "2", "--updates", "2", "--batch-size", "8", "--eval-every", "1"]
            for name, path in paths.items(): argv.extend(["--"+name.replace("_", "-"), str(path)])
            cache = lambda path: (None, train_records if str(path) == str(paths["train_cache"]) else dev_records)
            with patch.object(train, "FrozenV18", Provider), patch.object(train, "NuScenesWindowSource", lambda *a, **kw: SimpleNamespace(nusc=None)), \
                 patch.object(train, "load_manifest", return_value=(manifest, keys, None)), patch.object(train, "load_cache", side_effect=cache), \
                 patch.object(self.common, "gt_moving_support_sequence", return_value=moving):
                out = root / "first"
                with patch("sys.argv", argv + ["--out-dir", str(out)]): train.main()
                data = json.loads((out / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(data["successful_updates"], 2); self.assertEqual(data["dev_windows"], 2)
                self.assertTrue((out / "best.pt").is_file()); self.assertTrue((out / "last.pt").is_file())
                self.assertIn("last_learned_candidate", summarize(data))
                self.assertEqual(data["report"]["delta_vs_v18_pp"]["MovingMicro"], 0.)
                undefined = copy.deepcopy(data)
                undefined["report"]["delta_vs_v18_pp"]["MovingMicro"] = None
                undefined["report"]["delta_vs_v18_pp"]["per_horizon"]["1.0"]["MovingMicro"] = None
                self.assertIn("dMovingMicro=NA", summarize(undefined))
                resumed = root / "resumed"
                with patch("sys.argv", argv + ["--out-dir", str(resumed), "--resume", str(out / "last.pt")]): train.main()
                a = torch.load(out / "last.pt", weights_only=False)
                c = torch.load(resumed / "last.pt", weights_only=False)
                for key in a["state_dict"]: torch.testing.assert_close(a["state_dict"][key], c["state_dict"][key], rtol=0, atol=0)
                self.assertEqual(json.loads((resumed / "summary.json").read_text())["successful_updates"], 2)
            requests.clear()
            with patch.object(evaluation, "FrozenV18", Provider), patch.object(evaluation, "NuScenesWindowSource", lambda *a, **kw: None), \
                 patch.object(evaluation, "load_manifest", return_value=(manifest, keys[:2], None)), \
                 patch.object(evaluation, "load_cache", return_value=(None, dev_records)), \
                 patch("sys.argv", ["eval", "--config", str(config), "--val-cache", str(paths["dev_cache"]),
                    "--population-manifest", str(paths["population_manifest"]), "--base-checkpoint", str(paths["base_checkpoint"]),
                    "--selector-checkpoint", str(out / "best.pt"), "--dataroot", str(root), "--info-pkl", str(paths["dev_info"]),
                    "--out-dir", str(root / "predictions"), "--device", "cpu", "--predict-only"]):
                evaluation.main()
            self.assertEqual(requests, [False, False])
            files = list((root / "predictions").glob("*.npz")); self.assertEqual(len(files), 2)
            self.assertEqual(np.load(files[0])["predictions"].shape, (6, *grid.shape_hwd))
            self.assertFalse(json.loads((root / "predictions" / "summary.json").read_text())["future_gt_read"])


if __name__ == "__main__": unittest.main()
