"""Matched independent RANDOM GT-body TRAIN20% x1; baseline or unified-fit."""
import hashlib
from pathlib import Path
import signal
import sys
import threading
import numpy as np
from tools.joint_occ_experiments import run as legacy, data as old_data
from tools.shared_history_experiments.full_gt.model import GTBody
from tools.shared_history_experiments.full_gt import engine as gt
from tools.shared_history_experiments.run import NoLegacyHeadGraph
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.surface_ego_ablation_common import tensor_state_fingerprint
from tools.perception_robustness.run import stc_inventory
from .data import ARMS, ego_inputs, history_inputs, prefetch_adapter
from .evaluation import evaluator

PROTOCOL = 'matched_random_unified_track_KTA_train20pct_1epoch_v1'


def parser():
    p = legacy.parser(); p.description = __doc__
    p.set_defaults(mode='gt', epochs=1, train_fraction=.2, eval_population='dev64')
    p.add_argument('--track-arm', choices=ARMS, required=True); p.add_argument('--stc-root', required=True)
    return p


def validate(a):
    if (a.mode != 'gt' or a.epochs != 1 or a.train_fraction != .2 or a.eval_population != 'dev64'
            or a.window_batch_size != 4 or a.source_budget != 128 or a.seed != 20261011
            or a.training_backend != 'fused' or (a.object_weight, a.ego_weight, a.lovasz_weight) != (1., 1., 1.)):
        raise ValueError('fixed RANDOM GT-body TRAIN4086 x1 / B4-source128 / H6 full CE+Lovasz / DEV64')


def dependency_files():
    root = Path(__file__).resolve().parents[2]
    files = list(Path(__file__).parent.glob('*.py'))+list((root/'tools/joint_occ_experiments').glob('*.py'))
    files += [root/p for p in ('tools/unified_source_tracks/geometry.py',
        'tools/shared_history_experiments/full_gt/model.py', 'tools/shared_history_experiments/full_gt/engine.py',
        'tools/shared_history_experiments/refine/history.py', 'tools/shared_history_experiments/refine/model.py',
        'tools/shared_history_experiments/refine/geometry.py', 'tools/shared_history_experiments/model.py',
        'tools/shared_history_experiments/history.py', 'tools/final_semantic_evaluation/predictor.py',
        'tools/final_semantic_evaluation/metrics.py', 'real_motion/source_evidence_audit.py', 'real_motion/strong_w2det.py',
        'tools/real_motion/causal_column_common.py', 'tools/scene_ego_head/integration.py',
        'tools/scene_ego_head/diagnostics/engine.py', 'tools/perception_robustness/run.py',
        'tools/shared_history_experiments/actions/validation/workflow.py')]
    return sorted(set(files))


def main(event=None, argv=None):
    p = parser(); a = p.parse_args(argv); validate(a)
    out = Path(a.out_dir).resolve()
    if a.evaluate_only or a.resume and Path(a.resume).resolve().parent != out:
        raise ValueError('same-directory resume ONLY; no importing old trained weights')
    inventory = stc_inventory(a)
    names = ('parser', 'train_arm', 'evaluate', 'JointOccEgo', 'HistoryInputs', 'BothActionsExecution',
        'PROTOCOL', 'summarize', 'stable_json_fingerprint', 'save_resume_checkpoint', 'prefetched_batches')
    original = {n: getattr(legacy, n) for n in names}; old_ego = old_data.ego_inputs
    root = Path(__file__).resolve().parents[2]; files = dependency_files(); initializer = {}
    def model(config):
        value = GTBody(config); initializer['sha256'] = tensor_state_fingerprint(value.state_dict()); return value
    def fingerprint(value):
        if isinstance(value, dict) and any(str(k).replace('\\', '/') == 'tools/joint_occ_experiments/run.py' for k in value):
            value = dict(base=value, track_arm=a.track_arm, STC_inventory=inventory,
                prototype={str(f.relative_to(root)): legacy.sha256(f) for f in files})
        return original['stable_json_fingerprint'](value)
    def train(model, optimizer, provider, rows, mode, **kwargs):
        result = gt.train_arm(model, optimizer, provider, rows, mode, **kwargs)
        result['track_arm'] = a.track_arm
        if a.track_arm == 'unified_fit':
            result['track_valid_point_counts'] = np.sum([r[3][0]['track_fit']['valid_point_counts'] for r in rows], 0).tolist()
        return result
    def save(path, payload):
        payload = dict(payload, track_arm=a.track_arm, random_initializer=initializer['sha256'])
        original['save_resume_checkpoint'](path, payload)
        if Path(path).name == 'last.pt' and payload['epoch'] == 1 and payload['batch'] == 0:
            final = Path(path).with_name('epoch_0001.pt')
            if not final.exists(): original['save_resume_checkpoint'](final, payload)
    def summary(result):
        lines = ['===== MATCHED TRACK / ONE-EPOCH GT BODY =====',
            f'status={result["status"]}; arm={a.track_arm}; updates={result.get("updates", 0)}; epochs={result.get("epoch", 0)}',
            'Independent SAME RANDOM init / TRAIN4086 x1 / order / B4-source128 / H6 / full-cycle cosine.',
            'ONLY six-frame CE+Lovasz; no ego head, motion/existence loss, fitted prior or new bank.',
            'Baseline vs unified track + actual-time multi-frame KTA. Original Strong/CLEAR fixed.',
            'Same final DEV64 OCC+GT / STC+GT, full grid, STC full/all-known, no mask/alignment.',
            'Old checkpoints/caches read-only; changed derived track/registration/reads rebuilt in RAM.']
        if 'performance' in result: lines.append('PERFORMANCE '+legacy.json.dumps(result['performance']))
        for n, m in result.get('evaluation', {}).get('variants', {}).get('gt', {}).items():
            lines.append(f'{n} mIoU={m["mIoU"]:.6f} IoU={m["IoU"]:.6f} MovingMicro={m["MovingMicro"]:.6f}')
        return '\n'.join(lines)+'\n'
    try:
        legacy.parser = lambda: p
        legacy.train_arm, legacy.evaluate = train, evaluator(a.stc_root, a.track_arm, inventory=inventory)
        legacy.JointOccEgo, legacy.HistoryInputs = model, history_inputs(a.track_arm)
        legacy.BothActionsExecution, legacy.PROTOCOL = NoLegacyHeadGraph, PROTOCOL
        legacy.summarize, legacy.stable_json_fingerprint, legacy.save_resume_checkpoint = summary, fingerprint, save
        legacy.prefetched_batches = prefetch_adapter(original['prefetched_batches'], a.track_arm)
        old_data.ego_inputs = ego_inputs
        lock = out.parent/'.track_train_locks'/hashlib.sha256(str(out).encode()).hexdigest(); lock.mkdir(parents=True, exist_ok=True)
        with evaluation_lock(lock): return legacy.main(event, argv)
    finally:
        old_data.ego_inputs = old_ego
        for name, value in original.items(): setattr(legacy, name, value)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: event.set())
    sys.exit(main(event) or 0)
