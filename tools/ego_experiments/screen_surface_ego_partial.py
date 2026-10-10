#!/usr/bin/env python3
"""Train one pilot epoch from an immutable, already extracted TRAIN prefix."""
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from copy import deepcopy
import argparse
import gc
import json
import os
import signal
import threading
import time

import torch

from tools.ego_experiments import train_surface_ego_full as full
from tools.ego_experiments import eval_surface_ego_full as dev
from tools.real_motion.ego_trajectory_common import digest_file, fingerprint, row_fingerprint, implementation_fingerprint
from tools.real_motion.surface_ego_ablation_common import verify_sources, recover_original_initialization
from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
from tools.real_motion.waymo_zero_shot_common import write_json

FILES = ('tools/ego_experiments/screen_surface_ego_partial.py',
         'tools/real_motion/run_p0_f9_surface_ego_partial.sh')
PROTOCOL = 'surface_ego_existing_train_prefix_one_epoch_screen_v1'


def prefix_count(parent, total):
    for i in range(total):
        if not (Path(parent)/'bank'/f'{i:06d}.pt').is_file(): return i
    return total


def read_prefix(parent, c, count, *, expected=None, max_mib=4096, stop_event=None):
    if not 0 < count <= len(c['keys']): raise ValueError('bounded nonempty prefix required')
    sha = fingerprint(c); rows = []; receipt = []; size = 0
    for i in range(count):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('prefix load stopped; source intact')
        p = Path(parent)/'bank'/f'{i:06d}.pt'; before = digest_file(p)
        row = torch.load(p, map_location='cpu', weights_only=False)
        if (row.get('key') != c['keys'][i] or row.get('contract_fingerprint') != sha or
                row.get('content_fingerprint') != row_fingerprint(row) or
                row.get('origin') not in ('fresh_history', 'original_read_only_bank') or digest_file(p) != before):
            raise RuntimeError('parent bank identity/content changed: '+str(p))
        receipt.append(dict(index=i, sha256=before))
        size += sum(v.numel()*v.element_size() for v in row['features'].values())
        if size > max_mib*1024**2: raise RuntimeError('bounded prefix feature memory exceeded')
        rows.append(row)
    if expected is not None and receipt != expected: raise RuntimeError('frozen prefix receipt changed')
    return rows, receipt


def prefix_split(c, count):
    # Keep EXACT parent scene membership; never move held-out windows into fit.
    split = deepcopy(c['split'])
    split['train_indices'] = [i for i in split['train_indices'] if i < count]
    split['holdout_indices'] = [i for i in split['holdout_indices'] if i < count]
    if not split['train_indices'] or not split['holdout_indices']:
        raise ValueError('completed prefix needs both parent fit and held-out scenes')
    split['train_scenes'] = sorted({c['keys'][i][0] for i in split['train_indices']})
    split['holdout_scenes'] = sorted({c['keys'][i][0] for i in split['holdout_indices']})
    return split


def verify_parent(parent, c, sha, *, root=None):
    root = root or full.ROOT
    if digest_file(Path(parent)/'training.json') != sha or c['protocol'] != full.PROTOCOL or 'partial_screen' in c:
        raise RuntimeError('immutable full one-epoch parent required')
    if implementation_fingerprint(root) != c['feature_geometry_implementation']:
        raise RuntimeError('parent historical feature implementation changed')
    if {p:digest_file(Path(root)/p) for p in full.IMPLEMENTATION} != c['implementation']:
        raise RuntimeError('parent full training implementation changed; do not invalidate recovery')
    verify_sources(c['source'])


def verify_receipt(parent, receipt):
    for r in receipt:
        if digest_file(Path(parent)/'bank'/f'{r["index"]:06d}.pt') != r['sha256']:
            raise RuntimeError('parent prefix shard changed during pilot')


def fit(parent, out, *, max_windows=10240, min_windows=1024, resume=False, device='cuda',
        stop_event=None):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if parent == out or out.is_relative_to(parent) or parent.is_relative_to(out):
        raise ValueError('independent pilot output required')
    if max_windows < min_windows or min_windows < 2: raise ValueError('bounded positive prefix budget required')
    if out.exists() and not resume: raise FileExistsError('NEW pilot output or --resume SAME directory required')
    if resume and not (out/'training.json').is_file(): raise ValueError('pilot recovery contract missing')
    if device.startswith('cuda') and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    # Kernel lease refuses an active parent. Never stop/kill it automatically.
    with evaluation_lock(parent):
        started = time.perf_counter(); path = parent/'training.json'; sha = digest_file(path)
        c = json.loads(path.read_text(encoding='utf-8')); verify_parent(parent,c,sha)
        execution = dict(device=str(torch.device(device)),torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,WM_precision='bf16' if device.startswith('cuda') else 'fp32')
        if (execution != c['execution'] or
                {k:v for k,v in sorted(os.environ.items()) if k.startswith('SWFM_')} != c['runtime_environment']):
            raise RuntimeError('parent feature execution/environment required; use wrapper')
        previous = json.loads((out/'training.json').read_text(encoding='utf-8')) if resume else None
        count = previous['partial_screen']['windows'] if previous else min(prefix_count(parent,len(c['keys'])),max_windows)
        if not min_windows <= count <= max_windows: raise ValueError('not enough completed prefix; no new feature extraction')
        try:
            rows, receipt = read_prefix(parent,c,count,
                expected=previous['partial_screen']['shards'] if previous else None,stop_event=stop_event)
        except InterruptedError:
            print('Prefix load stopped safely; no source bank changes.'); return 130
        contract = deepcopy(c); contract['keys'] = c['keys'][:count]; contract['split'] = prefix_split(c,count)
        contract['selection'] = 'final pilot epoch1; incomplete scene-sorted TRAIN prefix; not representative full TRAIN'
        contract['partial_screen'] = dict(protocol=PROTOCOL,parent=str(parent),parent_contract_sha256=sha,
            windows=count,max_windows=max_windows,min_windows=min_windows,shards=receipt,
            implementation={p:digest_file(full.ROOT/p) for p in FILES},
            feature_extraction=False,full_continuation='ONLY causal bank reused; pilot Adam/weights NOT full resume')
        if previous is not None:
            if previous != contract: raise RuntimeError('pilot prefix/source/budget/code changed')
        else: out.mkdir(parents=True); write_json(out/'training.json',contract)
        print(f'EGO_EXISTING_PREFIX: {count}/{len(c["keys"])} cached windows, '
            f'fit={len(contract["split"]["train_indices"])} holdout={len(contract["split"]["holdout_indices"])}; '
            'NO feature extraction, one ego-head epoch, frozen WM/CCR.',flush=True)
        with evaluation_lock(out):
            source = c['source']; original_path = Path(source['source_dir'])/'head_last.pt'
            saved = torch.load(original_path,map_location='cpu',weights_only=False)
            if saved['contract'] != source['original_training'] or saved['config'] != c['head_config']:
                raise RuntimeError('original initializer source contract changed')
            head = recover_original_initialization(saved,source['frozen_checkpoint']).to(device)
            cached_seconds = time.perf_counter()-started; tick = time.perf_counter()
            kw = {k:c['schedule'][k] for k in ('epochs','batch_size','lr','min_lr','seed')}
            result = full.train(head,rows,contract,out,resume=resume and (out/'last.pt').is_file(),
                stop_event=stop_event,**kw)
            result['stage_seconds'] = dict(read_existing_bank_and_initializer=cached_seconds,
                ego_training_and_reports=time.perf_counter()-tick)
            result['seconds_this_invocation'] = time.perf_counter()-started
            verify_parent(parent,c,sha); verify_receipt(parent,receipt)
            if contract['partial_screen']['implementation'] != {p:digest_file(full.ROOT/p) for p in FILES}:
                raise RuntimeError('pilot implementation changed during run')
            inventory = dict(reused=sum(r['origin']=='original_read_only_bank' for r in rows),
                fresh=sum(r['origin']=='fresh_history' for r in rows))
            result['partial_population'] = dict(windows=count,parent_windows=len(c['keys']),
                scene_sorted_prefix_not_representative=True,no_new_feature_extraction=True)
            write_json(out/'training_summary.json',result)
            text = full.training_text(result,contract,inventory)+'Already-extracted scene-sorted prefix only; '
            text += 'NOT a representative full TRAIN run. No automatic full continuation or promotion.\n'
            (out/'summary.txt').write_text(text,encoding='utf-8'); print(text,flush=True)
    # A full-model dev eval is run by the wrapper only after completed training.
    del rows, head; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return 0 if result['status'] == 'complete' else 130


def main(stop_event=None, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--max-windows',type=int,default=10240);p.add_argument('--min-windows',type=int,default=1024)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a = p.parse_args(argv)
    return fit(a.parent_dir,a.out_dir,max_windows=a.max_windows,min_windows=a.min_windows,
        device=a.device,resume=a.resume,stop_event=stop_event)


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    sys.exit(main(event) or 0)
