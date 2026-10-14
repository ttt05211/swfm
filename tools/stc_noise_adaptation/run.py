"""Warm-start GT epoch2; frozen Encoder/STWM; matched Refiner-only 500 updates."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np
import torch
from tools.joint_occ_experiments import run as legacy, data as joint_data
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.loss_controls.run import capped_groups
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.perception_robustness.evaluation import evaluator
from tools.scene_ego_head.frozen_gt.run import validate_completed_source
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from .augmentation import load_model, RECIPE
from .data import history_inputs, prefetch_adapter, finish_backgrounds

PROTOCOL = 'STC_VAL_calibrated_history_GTepoch2_refiner_adapt500_v1'


class AdaptBody(GTBody):
    def __init__(self, config, source, digest):
        c = source['contract']['model']; shared, repair = c['shared_history'], c['repair']
        super().__init__(config, feature_width=shared['width'], z_bins=shared['z_bins'],
            memory_width=repair['memory_width'], width=repair['width'], chunk_size=repair['chunk_size'])
        self.load_state_dict(source['models']['gt'], strict=True)
        self.columns.epsilon = repair['epsilon']; self.source_sha256 = digest
        self.transport.eval().requires_grad_(False)

    def train(self, mode=True):
        super().train(mode); self.transport.eval(); return self

    def requires_grad_(self, value=True):
        super().requires_grad_(value); self.transport.requires_grad_(False); return self

    def configs(self):
        return {**super().configs(), 'adaptation': dict(protocol=PROTOCOL, fixed_updates=500,
            source_sha256=self.source_sha256, frozen=['SharedEncoder', 'STWM'], trainable=['SemanticRefiner'],
            initialization='GT_epoch2_weights_fresh_Adam_RNG', prior_coefficient=1.,
            training_known='dense_valid', new_feature_bank=False)}


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64',
                   motion_lr=1e-5, repair_lr=1e-4, ego_lr=1e-4)
    p.add_argument('--gt-checkpoint', required=True)
    p.add_argument('--noise-model', required=True)
    p.add_argument('--stc-root', required=True)
    p.add_argument('--augmentation', choices=('clean', 'empirical'), required=True)
    return p


def selection(a, noise):
    _, records = legacy.load_cache(a.dev_cache)
    eligible = [(str(r['scene_name']), str(r['t0_token'])) for r in records if str(r['scene_name']) in noise.remaining]
    if len(eligible) < 64 or len(set(eligible)) != len(eligible): raise RuntimeError('invalid remaining-scene population')
    keys = legacy.select_scene_balanced_round_robin(eligible, 64)
    selected = legacy.align_records(records, keys)
    inventory = []
    for record in selected:
        for token in record['history_tokens'][-4:]:
            path = Path(a.stc_root)/str(record['scene_name'])/str(token)/'labels.npz'
            st = path.stat()
            if not path.is_file() or not st.st_size: raise RuntimeError('missing selected STC history: '+str(path))
            inventory.append([str(path.resolve()), st.st_size, st.st_mtime_ns])
    return [list(k) for k in keys], inventory


def validate_args(a):
    if (a.mode != 'gt' or a.epochs != 1 or a.train_fraction != .2 or a.eval_population != 'dev64'
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011
            or a.training_backend != 'fused' or not 0 <= a.max_updates <= 500
            or (a.motion_lr, a.repair_lr, a.ego_lr) != (1e-5, 1e-4, 1e-4)
            or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)
            or a.evaluate_only):
        raise ValueError('fixed warm GTepoch2 Refiner-only / TRAIN20% partial500 / B4 / H6 / CE+Lovasz / remaining64')


def main(event=None, argv=None):
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
             'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'epoch_groups', 'write_json',
             'restore_checkpoint', 'load_manifest', 'prefetched_batches')
    old = {n: getattr(legacy, n) for n in names}; old_ego = joint_data.ego_inputs
    p = parser(); a = p.parse_args(argv); validate_args(a)
    source_path, out = Path(a.gt_checkpoint).resolve(), Path(a.out_dir).resolve()
    if out == source_path.parent or source_path.parent in out.parents: raise ValueError('output outside source GT directory required')
    if a.resume and Path(a.resume).resolve() != out/'train/last.pt': raise ValueError('resume SAME arm train/last.pt only')
    noise, noise_identity = load_model(a.noise_model); keys, inventory = selection(a, noise)
    if legacy.sha256(source_path) != noise_identity['source_GT_sha256']: raise RuntimeError('profile belongs to another GT body')
    lock = out.parent/'.empirical_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
    with evaluation_lock(lock):
        if not a.resume:
            out.mkdir(parents=True, exist_ok=False); (out/'frozen').mkdir()
            snapshot_checkpoint(source_path, out/'frozen/GTepoch2.pt')
        frozen = out/'frozen/GTepoch2.pt'; digest = legacy.sha256(frozen)
        if digest != noise_identity['source_GT_sha256']: raise RuntimeError('frozen GT source changed')
        source = torch.load(frozen, map_location='cpu', weights_only=False); validate_completed_source(source)
        for k in ('train_cache', 'dev_cache', 'train_info', 'dev_info', 'base_checkpoint'):
            if legacy.sha256(getattr(a, k)) != source['contract']['data'][k]: raise RuntimeError('GT source metadata differs: '+k)
        frozen_digest = []
        def construct(config):
            m = AdaptBody(config, source, digest)
            frozen_digest[:] = [m, tensor_state_fingerprint(m.transport.state_dict())]
            old['write_json'](out/'initialization_audit.json', dict(source_sha256=digest,
                Encoder_STWM_sha256=frozen_digest[1], Refiner_sha256=tensor_state_fingerprint(m.columns.state_dict()),
                imported_Adam=False, imported_RNG=False, ego_head=False))
            return m
        root = Path(__file__).resolve().parents[2]
        dependencies = ('tools/shared_history_experiments/full_gt/model.py', 'tools/shared_history_experiments/full_gt/engine.py',
            'tools/shared_history_experiments/refine/history.py', 'tools/shared_history_experiments/refine/model.py',
            'tools/shared_history_experiments/refine/geometry.py', 'tools/shared_history_experiments/model.py',
            'tools/shared_history_experiments/history.py', 'tools/perception_robustness/data.py',
            'tools/perception_robustness/evaluation.py', 'tools/final_semantic_evaluation/predictor.py',
            'tools/scene_ego_head/integration.py', 'tools/scene_ego_head/diagnostics/engine.py',
            'tools/history_validity/data.py', 'tools/stc_noise_profile/statistics.py', 'tools/stc_noise_profile/data.py')
        files = list(Path(__file__).parent.glob('*.py'))+[root/x for x in dependencies]
        def fingerprint(value):
            if isinstance(value, dict) and 'tools/joint_occ_experiments/run.py' in value:
                value = dict(base=value, prototype={str(f.relative_to(root)): legacy.sha256(f) for f in files})
            return old['stable_json_fingerprint'](value)
        def mark(contract):
            contract.update(initialization='GT_epoch2_fresh_Adam_RNG', source_GT_sha256=digest,
                augmentation=a.augmentation, noise_identity=noise_identity, recipe=RECIPE,
                eval_selection='fixed_remaining64_excluded_from_noise_fit_NOT_pristine_test',
                training_known='dense_valid_BOTH', OCC_eval_known='original_lidar_known',
                objective={'CE': 1., 'Lovasz': 1., 'motion': 0., 'existence': 0., 'ego': 0.},
                action_thresholds_used=False, thresholds=None, weights=[0., 0., 1.], complete_dataset_epochs=0)
        def write(path, value):
            if Path(path).name == 'contract.json': mark(value)
            return old['write_json'](path, value)
        def restore(saved, models, optimizers, contract):
            mark(contract); return old['restore_checkpoint'](saved, models, optimizers, contract)
        def manifest(path):
            original, _, third = old['load_manifest'](path)
            return original, list(map(tuple, keys)), third
        def train(model, optimizer, provider, rows, mode, **kw):
            tick = time.perf_counter(); finished = [finish_backgrounds(provider, r) for r in rows]
            seconds = time.perf_counter()-tick
            kw['probe'] = False  # frozen F/XY are not autograd roots
            stats = gt.train_arm(model, optimizer, provider, finished, mode, **kw)
            stats['seconds'] += seconds; stats['stages_seconds']['fresh_corrupted_Strong'] = seconds
            audits = [r[1].get('_empirical_noise', {}) for r in rows]
            stats['augmentation'] = dict(arm=a.augmentation, selected_windows=sum(bool(v.get('selected')) for v in audits),
                changed_windows=sum(not v.get('clean_geometry_reusable', True) for v in audits),
                changed_voxels=sum(sum(v.get('changed_voxels', [])) for v in audits),
                probability_clipped_cells=sum(v.get('probability_clipped_cells', 0) for v in audits),
                CPU_simulation_seconds_sum=sum(v.get('simulation_seconds', 0.) for v in audits),
                CPU_geometry_seconds_sum=sum(v.get('geometry_prepare_seconds', 0.) for v in audits),
                generated_errors={f: sum(sum(v.get('generated_errors', {}).get(f, [])) for v in audits)
                                  for f in ('false_occupancy', 'missed_occupancy', 'wrong_class')},
                radial_errors={f: np.sum([v.get('radial_errors', {}).get(f, [0]*6) for v in audits], axis=0).tolist()
                               for f in ('false_occupancy', 'missed_occupancy', 'wrong_class')},
                radial_opportunities={f: np.sum([v.get('radial_opportunities', {}).get(f, [0]*6) for v in audits], axis=0).tolist()
                                      for f in ('GT_free', 'GT_occupied')},
                actual_sources=sum(len(r[0]['source_class_id']) for r in finished),
                nominal_sources=sum(v.get('original_sources', len(r[0]['source_class_id'])) for r, v in zip(rows, audits)))
            return stats
        eval_base = evaluator(a.stc_root, inventory=inventory)
        def evaluate(models, *args, **kw):
            if tensor_state_fingerprint(models['gt'].transport.state_dict()) != frozen_digest[1]: raise RuntimeError('frozen body changed')
            r = eval_base(models, *args, **kw)
            if r['windows'] != 64: raise RuntimeError('exact remaining64 required')
            r['scoring'] = 'fixed remaining64; full-grid; OCC original lidar-known / STC dense-valid; GT future ego'
            r['evaluation_keys'] = keys; return r
        def summary(r):
            lines = ['===== VAL-CALIBRATED HISTORY / REFINER ADAPT500 =====', 'status='+r['status'],
                f'arm={a.augmentation}; updates={r.get("updates", 0)}/500; complete_dataset_epochs=0',
                'Same GTepoch2 weights/fresh Adam/order/B4/H6; Encoder/STWM frozen; ONLY Refiner CE+Lovasz.',
                'Both TRAIN dense-valid; empirical arm 50% correlated historical corruption; future targets clean.',
                'prior c=1 / epsilon unchanged; no new gate/ego/motion/existence objectives.',
                'Fixed remaining64 EXCLUDED from this VAL noise fit, NOT a pristine independent test or formal zero-shot.',
                'Not original DEV64/DEV512. Actual changed histories rebuild source/KTA/Strong/prior/memory.']
            if 'performance' in r: lines += ['PERFORMANCE '+legacy.json.dumps(r['performance'])]
            for name, m in r.get('evaluation', {}).get('variants', {}).get('gt', {}).items():
                lines.append(f'{name} mIoU={m["mIoU"]:.6f} IoU={m["IoU"]:.6f} MovingMicro={m["MovingMicro"]:.6f}')
            if r.get('error'): lines += ['error='+r['error']]
            return '\n'.join(lines)+'\n'
        args = sys.argv[1:] if argv is None else list(argv)
        args = list(args); args[args.index('--out-dir')+1] = str(out/'train')
        try:
            legacy.parser = lambda: p
            legacy.JointOccEgo, legacy.HistoryInputs = construct, history_inputs(keys)
            legacy.BothActionsExecution, legacy.PROTOCOL = NoLegacyHeadGraph, PROTOCOL
            legacy.train_arm, legacy.evaluate, legacy.summarize = train, evaluate, summary
            legacy.epoch_groups = lambda *xs: capped_groups(old['epoch_groups'], *xs)
            legacy.stable_json_fingerprint, legacy.write_json, legacy.restore_checkpoint = fingerprint, write, restore
            legacy.load_manifest = manifest
            legacy.prefetched_batches = prefetch_adapter(old['prefetched_batches'], a.augmentation, noise, a.seed)
            joint_data.ego_inputs = gt.history_inputs
            print(f'EMPIRICAL500 arm={a.augmentation}; warm GTepoch2 / frozen Encoder-STWM / H6 / Refiner-only.', flush=True)
            print('The shared driver may print RANDOM/epoch=1; this experiment is WARM initialization and a 500-step PARTIAL pass.', flush=True)
            status = legacy.main(event, args)
            r = legacy.json.loads((out/'train/results.json').read_text(encoding='utf-8'))
            old['write_json'](out/'results.json', r); (out/'summary.txt').write_text(summary(r), encoding='utf-8')
            return status
        finally:
            joint_data.ego_inputs = old_ego
            for n, v in old.items(): setattr(legacy, n, v)
            if frozen_digest:
                m, before = frozen_digest; unchanged = tensor_state_fingerprint(m.transport.state_dict()) == before
                legacy.write_json(out/'freeze_audit.json', dict(Encoder_STWM_unchanged=unchanged,
                    frozen_parameter_gradients=sum(p.grad is not None for p in m.transport.parameters())))
                if not unchanged: raise RuntimeError('frozen Encoder/STWM mutated')


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
