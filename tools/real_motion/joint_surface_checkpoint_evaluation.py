"""Shared-history, window-major Surface CCR comparison with exact integer resume."""
from collections import defaultdict
import time

import numpy as np
import torch

from real_motion.column_runtime_pipeline import prefetch_raw_columns
from real_motion.source_evidence_audit import edit_quality
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion import causal_column_common as columns
from tools.real_motion import ccr_screen_common as ccr
from tools.real_motion.eval_p0_f9_v17_local_stwm import window_from_record
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import Metrics
from tools.real_motion.joint_surface_ccr_common import prepare_live
from tools.real_motion.shared_evidence_pilot_common import moving_support_masks

VARIANTS = ('static_repair', 'dynamic_repair', 'joint')
COUNT_FIELDS = ('oi', 'ou', 'si', 'su', 'mi', 'mu')


def dump_metric(metric):
    return {k: getattr(metric, k).tolist() for k in COUNT_FIELDS}


def load_metric(row):
    if set(row) != set(COUNT_FIELDS):
        raise RuntimeError('saved metric fields changed')
    result = Metrics()
    for key in COUNT_FIELDS:
        value = np.asarray(row[key])
        if value.dtype.kind not in 'iu' or value.shape != getattr(result,key).shape or np.any(value < 0):
            raise RuntimeError('invalid saved integer metric counts')
        getattr(result,key)[:] = value
    for first, second in (('oi','ou'),('si','su'),('mi','mu')):
        if np.any(getattr(result,first)>getattr(result,second)):
            raise RuntimeError('intersection exceeds union')
    return result


class Accumulator:
    def __init__(self):
        self.cursor = 0; self.base = Metrics()
        self.metrics = {k:Metrics() for k in VARIANTS}
        self.quality = {k:defaultdict(int) for k in VARIANTS}
        self.scenes = defaultdict(lambda:{k:Metrics() for k in ('baseline',*VARIANTS)})
        self.action = np.zeros((2,2,4),np.int64)

    def update(self, record, prep, evidence, plan, probabilities, moving):
        raw = prep.raw; target, valid = ccr.repair_targets(evidence,plan,raw['future_gt_occ'])
        predicted = np.zeros_like(plan.legal,dtype=bool)
        predicted[...,0] = (probabilities[...,0]>=.5) & plan.legal[...,0]
        for role in (0,1):
            for action in (0,1):
                mask = valid[...,action] & ((evidence.actor>=0)==bool(role))[:,None]
                y = target[...,action] & mask; z = predicted[...,action] & mask
                self.action[role,action] += (int((y&z).sum()),int((~y&z&mask).sum()),
                                            int((y&~z).sum()),int(mask.sum()))
        remove = np.zeros_like(probabilities[...,1])
        predictions = {name:ccr.compose_canonical(prep.baseline,evidence,plan,
            probabilities[...,0],remove,thresholds=(.5,None),
            role={'static_repair':'static','dynamic_repair':'dynamic','joint':'all'}[name])
            for name in VARIANTS}
        scene = self.scenes[str(record['scene_name'])]
        for ri,h in enumerate(columns.REPORT):
            gt = raw['future_gt_occ'][h]; before = prep.baseline[h]
            counts = Metrics.counts(before,gt,moving[h],17)
            self.base.update(ri,counts=counts); scene['baseline'].update(ri,counts=counts)
            for name in VARIANTS:
                dense = predictions[name][h]; counts = Metrics.counts(dense,gt,moving[h],17)
                self.metrics[name].update(ri,counts=counts); scene[name].update(ri,counts=counts)
                for key,value in edit_quality(before,dense,gt).items():
                    self.quality[name][key] += value
        self.cursor += 1

    def dump(self):
        return dict(cursor=self.cursor,base=dump_metric(self.base),
            metrics={n:dump_metric(m) for n,m in self.metrics.items()},
            quality={n:dict(v) for n,v in self.quality.items()},action=self.action.tolist(),
            scenes={s:{n:dump_metric(m) for n,m in rows.items()} for s,rows in self.scenes.items()})

    @classmethod
    def restore(cls, saved, cursor):
        result = cls()
        if saved.get('cursor') != cursor or set(saved['metrics']) != set(VARIANTS) or set(saved['quality']) != set(VARIANTS):
            raise RuntimeError('saved all-model cursor/variants changed')
        result.cursor = cursor; result.base = load_metric(saved['base'])
        result.metrics = {n:load_metric(m) for n,m in saved['metrics'].items()}
        for name,row in saved['quality'].items():
            if any(type(v) is not int or v<0 for v in row.values()):
                raise RuntimeError('invalid saved edit counts')
            result.quality[name].update(row)
        action = np.asarray(saved['action'])
        if action.shape != (2,2,4) or action.dtype.kind not in 'iu' or (action<0).any():
            raise RuntimeError('invalid saved action counts')
        result.action[:] = action
        for scene,rows in saved['scenes'].items():
            if set(rows) != {'baseline',*VARIANTS}:
                raise RuntimeError('saved scene variants changed')
            result.scenes[scene] = {n:load_metric(m) for n,m in rows.items()}
        for name in ('baseline',*VARIANTS):
            aggregate = result.base if name=='baseline' else result.metrics[name]
            for key in COUNT_FIELDS:
                total = sum((getattr(rows[name],key) for rows in result.scenes.values()),
                            np.zeros_like(getattr(aggregate,key)))
                if not np.array_equal(total,getattr(aggregate,key)):
                    raise RuntimeError('scene counts differ from aggregate')
        return result

    def report(self):
        result = columns.report_states(self.base,self.metrics,self.quality,self.scenes)
        actions = {}
        for role,name in ((0,'static'),(1,'dynamic')):
            for action,label in ((0,'ADD'),(1,'REMOVE')):
                tp,fp,fn,valid = map(int,self.action[role,action])
                actions[name+'/'+label] = dict(tp=tp,fp=fp,fn=fn,valid=valid,
                    precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)
        result.update(windows=self.cursor,scenes=len(self.scenes),action_learning=actions,
                      support=ccr.SUPPORT_NOTE,decision_rule='weighted ADD raw0.5 / REMOVEoff')
        return result


@torch.no_grad()
def evaluate_group(provider, source, records, models, *, saved=None, start_window=0,
                   save_state=None, progress=None, stop_event=None, checkpoint_every=8):
    if not models or not 0<=start_window<=len(records) or checkpoint_every<1:
        raise ValueError('invalid comparison jobs/cursor')
    if saved is not None and set(saved)!=set(models):
        raise RuntimeError('saved model group changed')
    if start_window and saved is None:
        raise RuntimeError('missing resumed integer states')
    configs = {stable_json_fingerprint(m.configs()) for m in models.values()}
    if len(configs)!=1:
        raise RuntimeError('shared geometry requires identical model configurations')
    for model in models.values(): model.eval().requires_grad_(False)
    states = {n:Accumulator.restore(saved[n],start_window) if saved is not None else Accumulator() for n in models}
    checked = {n:False for n in models}
    totals = dict(raw_windows=0,moving_support_windows=0,model_windows=0,input_wait_seconds=0.,
                  moving_support_seconds=0.,model_seconds={n:0. for n in models})
    if save_state: save_state(start_window,{n:s.dump() for n,s in states.items()},totals)
    previous = time.perf_counter()
    with ccr._evaluation_geometry(provider,False):
        iterator = prefetch_raw_columns(provider,source,records[start_window:])
        try:
            for wi,(record,raw) in enumerate(iterator,start_window+1):
                if stop_event is not None and stop_event.is_set():
                    raise InterruptedError('stopped at all-model window boundary')
                started = time.perf_counter(); totals['input_wait_seconds'] += started-previous
                if raw is None: raise RuntimeError('missing shared history')
                tick = time.perf_counter(); window = window_from_record(record)
                support = ccr.gt_moving_support_sequence(source.nusc,window.t0_token,window.future_tokens,
                    tuple(.5*(h+1) for h in range(6)),grid=provider.pcfg.grid,workers=provider.workers)
                moving = moving_support_masks(support,provider.pcfg.grid.shape_hwd)
                totals['moving_support_seconds'] += time.perf_counter()-tick
                totals['raw_windows'] += 1; totals['moving_support_windows'] += 1
                for name,model in models.items():
                    tick = time.perf_counter()
                    provider.joint,provider.model = model,model.transport
                    provider.columns_checked = checked[name]
                    output = model.motion(record,provider.device)
                    # Each model gets new hard geometry/poses and six readouts.
                    # Only raw history and immutable cached support are shared.
                    local_raw = dict(raw)
                    prep = prepare_live(provider,record,local_raw,output)
                    checked[name] = bool(provider.columns_checked)
                    evidence = ccr.build_inputs(provider,prep)
                    plan = ccr.map_inputs(provider,evidence,prep)
                    probability = ccr.probabilities(model.columns,evidence,plan,output,provider.device)
                    states[name].update(record,prep,evidence,plan,probability,moving)
                    # Atlas is deterministic history-only; learned arrays never
                    # enter this shared dictionary or a persistent cache.
                    if '_surface_atlas_live' in local_raw:
                        raw['_surface_atlas_live'] = local_raw['_surface_atlas_live']
                    ccr.sync(provider.device)
                    seconds = time.perf_counter()-tick
                    totals['model_seconds'][name] += seconds; totals['model_windows'] += 1
                    if progress: progress(dict(event='candidate_window',checkpoint=name,window=wi,seconds=seconds))
                    del prep,evidence,plan,probability,output,local_raw
                stopping = stop_event is not None and stop_event.is_set()
                if save_state and (wi%checkpoint_every==0 or wi==len(records) or stopping):
                    save_state(wi,{n:s.dump() for n,s in states.items()},totals)
                if progress: progress(dict(event='all_model_window',window=wi,windows=len(records),
                    models=len(models),seconds=time.perf_counter()-started))
                if stopping: raise InterruptedError('stopped after complete all-model window')
                previous = time.perf_counter()
        finally:
            iterator.close()
    return {n:s.report() for n,s in states.items()},totals
