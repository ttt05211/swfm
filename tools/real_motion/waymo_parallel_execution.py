"""Bounded spawn workers; ONLY ordered integer counts cross process boundaries.

Independent CUDA models avoid shared streams/graph inputs/model callbacks.
Every worker finishes SIX forecasts before accessing future occupancy. The
parent owns one contiguous metric cursor; at most `processes` chunks in flight.
"""
import atexit
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from real_motion.waymo_i2world import WaymoMetrics, file_sha256, fingerprint
from real_motion.waymo_i2world_10hz import WaymoI2World10HzSource
from real_motion.waymo_geometry_execution import GeometryPrefetchSource
from real_motion.native_column_cpu import prepare_native
from real_motion.waymo_native_execution import prepare_waymo_native
from real_motion.strong_majority_execution import ParallelNativeMajority, strong_majority_execution
from tools.real_motion.joint_surface_checkpoint_selection import AVERAGE_EPOCHS, load_evaluation_model
from tools.real_motion.joint_surface_long_rollout_common import verify_first_block
from tools.real_motion.waymo_fast_execution import FastWaymoSurfaceProvider, FastSurfaceBlockExecution
from tools.real_motion.waymo_fast_execution_v2 import FastV2WaymoProvider, FastV2SurfaceExecution
from tools.real_motion.waymo_zero_shot_common import BRANCHES, restore

PROTOCOL = 'waymo_exact_bounded_spawn_counts_v1'
_worker = None


def bytes_digest(arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.asarray(a); h.update(str(a.shape).encode()); h.update(str(a.dtype).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _close_worker():
    global _worker
    if _worker is None: return
    _worker['execution'].close()
    if hasattr(_worker['provider'],'close'): _worker['provider'].close()
    if _worker['majority'] is not None: _worker['majority'].close()
    _worker=None


def _initialize(spec, barrier, slot_counter):
    """Trusted metadata/weights only, no raw GT or forecasts during startup."""
    global _worker
    signal.signal(signal.SIGINT,signal.SIG_IGN)  # parent drains whole bounded chunks
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    with slot_counter.get_lock():
        slot=slot_counter.value; slot_counter.value+=1
    root=Path(__file__).resolve().parents[2]
    if any(file_sha256(root/name)!=value for name,value in spec.get('implementation',{}).items()):
        raise RuntimeError('parallel worker implementation changed')
    if file_sha256(spec['checkpoint'])!=spec['checkpoint_sha256']:
        raise RuntimeError('parallel frozen checkpoint changed')
    source=WaymoI2World10HzSource.from_files(spec['waymo_root'],info_file=spec.get('info_file'),
        pose_file=spec.get('pose_file'),raw_free_label=spec['raw_free_label'],
        cache_mib=spec['frame_cache_mib'],shape=tuple(spec['shape']))
    selected=source.windows[:spec['windows']]; inventory=source.preflight(selected)
    if (source.manifest_fingerprint!=spec['manifest_fingerprint'] or source.metadata!=spec['data']
            or inventory!=spec['inventory']):
        raise RuntimeError('parallel data/population mismatch')
    saved,joint=load_evaluation_model(spec['checkpoint'],device=spec['device'],z_bins=spec['shape'][2])
    if (saved.get('source_epochs')!=list(AVERAGE_EPOCHS) or not saved.get('averaging')
            or joint.transport.config.history_frames!=4
            or file_sha256(spec['checkpoint'])!=spec['checkpoint_sha256']):
        raise RuntimeError('parallel FOUR-history frozen mean required')
    prepare_native(); prepare_waymo_native()
    v2=spec['backend']=='v2'
    provider=(FastV2WaymoProvider if v2 else FastWaymoSurfaceProvider)(joint,spec['pcfg'],spec['device'],
        spec['workers'],geometry_mib=spec['geometry_mib'])
    execution=(FastV2SurfaceExecution if v2 else FastSurfaceBlockExecution)(provider,workers=spec['workers'],
        query_workers=spec['workers'],graphs=spec['graphs'],surface_chunk=spec['surface_chunk'])
    majority=ParallelNativeMajority(min(4,spec['workers'])) if spec['parallel_majority'] else None
    _worker=dict(spec=spec,source=source,provider=provider,execution=execution,majority=majority,
                 checked=False,barrier=barrier,slot=slot)
    atexit.register(_close_worker)
    barrier.wait(timeout=120)  # both models fully ready before any timed task


@torch.no_grad()
def _chunk(task):
    w=_worker; spec=w['spec']; source=w['source']; provider=w['provider']; rows=[]
    if task.get('reset'):
        provider.geometry.clear(); source.cache.clear(); source.cache_bytes=0
        # One actual first-history window per worker, NOT all-window warm cache.
        indices=[task['warm_indices'][w['slot']]]
    else: indices=task['indices']
    selected=[source.windows[i] for i in indices]
    streaming=GeometryPrefetchSource(source,selected,provider.geometry,enabled=spec['history_prefetch'])
    try:
        for index,window in zip(indices,selected):
            tick=time.perf_counter(); mark=tick
            record,raw=streaming.prediction_inputs(window)
            stages=dict(history_io=time.perf_counter()-mark); mark=time.perf_counter()
            with strong_majority_execution(w['majority']) if w['majority'] is not None else nullcontext():
                prep=provider.prepare_columns(None,record,include_gt=False,raw_window=raw)
            model_stages=dict(history_and_transport_prepare=time.perf_counter()-mark,
                **{'prepare.'+k:v for k,v in provider.fast_prepare_stages.items()})
            dense,edits,timing,scores=w['execution'].predict(prep); model_stages.update(timing)
            if not w['checked']:
                verify_first_block(provider,prep.state['rec'],prep,dense,scores,w['execution']); w['checked']=True
            for forecasts in (prep.baseline,dense):
                if len(forecasts)!=6 or any(np.asarray(a).shape!=source.shape for a in forecasts):
                    raise RuntimeError('parallel incomplete SIX prediction before GT access')
            stages['prediction_and_first_exactness']=time.perf_counter()-mark; mark=time.perf_counter()
            # Diagnostic hashes are outside timed work, not used to report FPS.
            signatures=None
            if task.get('hashes'):
                signatures=dict(transport=bytes_digest(prep.baseline),dense=bytes_digest(dense),
                    probability=bytes_digest([scores]),motion=bytes_digest([
                        prep.state['rec'][k].cpu().numpy() for k in ('features','local_semantic_tube',
                        'kta_displacement_xy_m','frame_motion_features','target_source_mask_tube')]))
            hash_seconds=time.perf_counter()-mark; mark=time.perf_counter()
            targets=streaming.metric_targets(window)
            stages['future_target_io']=time.perf_counter()-mark; mark=time.perf_counter()
            counts={k:WaymoMetrics() for k in BRANCHES}
            counts['transport'].add(prep.baseline,targets); counts['joint'].add(dense,targets)
            if edits.get('removed',0): raise RuntimeError('parallel ADD-only forecast removed voxels')
            stages['integer_metrics']=time.perf_counter()-mark
            stages['window']=time.perf_counter()-tick-hash_seconds
            rows.append(dict(index=index,anchor=window.anchor,t0=record['t0_token'],pid=os.getpid(),
                counts={k:v.counts.tolist() for k,v in counts.items()},edits=edits,stages=stages,
                model_stages=model_stages,exactness_passed=w['checked'],signatures=signatures,
                geometry_cache=provider.geometry.stats()))
            del prep,dense,scores,targets
    finally: streaming.close()
    if task.get('reset'): w['barrier'].wait(timeout=120)
    return rows


class SpawnPool:
    def __init__(self,spec,processes):
        if not 1<=processes<=4 or not 1<=spec['workers']<=4:
            raise ValueError('bounded 1..4 processes / per-process threads required')
        self.spec=spec; self.processes=processes; self.ctx=mp.get_context('spawn')
        self.barrier=self.ctx.Barrier(processes); self.slots=self.ctx.Value('i',0)
        self.executor=ProcessPoolExecutor(max_workers=processes,mp_context=self.ctx,
            initializer=_initialize,initargs=(spec,self.barrier,self.slots))

    def reset(self,indices,chunk):
        warm=[indices[min(i*chunk,len(indices)-1)] for i in range(self.processes)]
        futures=[self.executor.submit(_chunk,dict(reset=True,warm_indices=warm)) for _ in range(self.processes)]
        rows=[f.result() for f in futures]
        if len({r[0]['pid'] for r in rows})!=self.processes:
            raise RuntimeError('parallel workers not independently warmed')

    def batches(self,indices,chunk,*,hashes=False,stop_event=None):
        if not 1<=chunk<=16: raise ValueError('chunk must be 1..16')
        chunks=(list(indices[b:b+chunk]) for b in range(0,len(indices),chunk))
        pending=deque()
        def submit():
            if stop_event is not None and stop_event.is_set(): return False
            value=next(chunks,None)
            if value is None: return False
            pending.append((value,self.executor.submit(_chunk,dict(indices=value,hashes=hashes))))
            return True
        for _ in range(self.processes): submit()
        while pending:
            expected,future=pending.popleft(); rows=future.result()
            if [r['index'] for r in rows]!=expected:
                raise RuntimeError('parallel results missing/duplicated/out of order')
            yield rows
            submit()  # interrupted: drain existing whole chunks, NO new work

    def close(self): self.executor.shutdown(wait=True,cancel_futures=True)


def evaluate_parallel(pool,indices,contract,*,shape,saved=None,save=None,progress=None,
                      stop_event=None,checkpoint_every=8,chunk=8):
    state=(dict(completed_windows=0,counts={k:WaymoMetrics().counts.tolist() for k in BRANCHES},
                exactness_passed=False,stage_seconds={},edits={}) if saved is None else
           restore(saved,contract,voxel_count=int(np.prod(shape))))
    if contract['windows']!=len(indices): raise ValueError('parallel population mismatch')
    metrics={k:WaymoMetrics(v,state['completed_windows']) for k,v in state['counts'].items()}
    stages=defaultdict(float,state['stage_seconds']); edits=defaultdict(int,state['edits'])
    cursor=state['completed_windows']; started=last=time.perf_counter(); volume=int(np.prod(shape))
    def persist():
        state.update(counts={k:v.counts.tolist() for k,v in metrics.items()},stage_seconds=dict(stages),
                     edits=dict(edits),contract_fingerprint=fingerprint(contract))
        value=dict(state); value['fingerprint']=fingerprint(value)
        if save: save(value)
    persist()
    try:
        for rows in pool.batches(indices[cursor:],chunk,stop_event=stop_event):
            now=time.perf_counter(); batch_wall=now-last; last=now
            # Validate the entire chunk before admitting ANY integer count.
            deltas=[]
            for offset,row in enumerate(rows):
                if row['index']!=indices[state['completed_windows']+offset] or not row['exactness_passed']:
                    raise RuntimeError('parallel unverified/wrong window')
                if set(row['counts'])!=set(BRANCHES) or row['edits'].get('removed',0):
                    raise RuntimeError('parallel invalid branch/ADD-only counts')
                item={}
                for key in BRANCHES:
                    a=np.asarray(row['counts'][key])
                    if (a.shape!=(3,18,18) or a.dtype.kind not in 'ui' or (a<0).any()
                            or not (a.sum((1,2))==volume).all()):
                        raise RuntimeError('parallel malformed one-window integer counts')
                    item[key]=a.astype(np.int64)
                for d in (row['stages'],row['model_stages'],row['edits']):
                    if any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<0 for v in d.values()):
                        raise RuntimeError('parallel invalid timing/edit value')
                deltas.append(item)
            for row,item in zip(rows,deltas):
                for key in BRANCHES: metrics[key].counts+=item[key]; metrics[key].windows+=1
                for k,v in row['stages'].items(): stages['parallel.worker.'+k]+=v
                for k,v in row['model_stages'].items(): stages['parallel.worker.model.'+k]+=v
                for k,v in row['edits'].items(): edits[k]+=int(v)
                state['completed_windows']+=1; state['exactness_passed']=True
                if progress:
                    progress(dict(window=state['completed_windows'],windows=len(indices),anchor=row['anchor'],
                        t0=row['t0'],worker_pid=row['pid'],worker_seconds=row['stages']['window'],
                        seconds=batch_wall/len(rows),stages=row['stages'],model_stages=row['model_stages'],
                        geometry_cache=row['geometry_cache'],edits=row['edits'],
                        parallel_elapsed_seconds=now-started,parallel_completed=state['completed_windows']-cursor,
                        timing_note='seconds is ordered parent batch wall/window; worker stages OVERLAP; NOT FPS'))
                if state['completed_windows']%checkpoint_every==0: persist()
    finally:
        stages['parallel.master_wall_seconds']+=time.perf_counter()-started; persist()
    return dict(status='complete' if state['completed_windows']==len(indices) else 'stopped',
        completed_windows=state['completed_windows'],reports={k:v.report() for k,v in metrics.items()},
        stages=dict(stages),edits=dict(edits),exactness_passed=state['exactness_passed'])


def paired_parallel_speed(spec,indices,*,processes=2,chunk=8,repeats=2,stop_event=None):
    """Same native indices/SIX bytes/counts, fixed total thread budget, real eval."""
    if len(indices)<processes or repeats<1: raise ValueError('speed needs >=processes windows')
    arms=dict(fast_v1_serial=(dict(spec,backend='v1',workers=spec['workers']*processes),1),
              fast_v2_parallel=(dict(spec,backend='v2'),processes))
    if arms['fast_v1_serial'][0]['workers']>4: raise ValueError('speed total thread budget must be <=4')
    pools={}; startup={}; durations=defaultdict(list)
    try:
        for name,(settings,n) in arms.items():
            tick=time.perf_counter(); pool=SpawnPool(settings,n); pools[name]=pool
            pool.reset(indices,chunk); startup[name]=time.perf_counter()-tick
        signatures={}; worker_pids={}
        for name,pool in pools.items():
            rows=[r for batch in pool.batches(indices,chunk,hashes=True,stop_event=stop_event) for r in batch]
            if len(rows)!=len(indices): raise InterruptedError('parallel speed stopped')
            signatures[name]=[(r['index'],r['signatures'],r['counts']) for r in rows]
            worker_pids[name]=sorted({r['pid'] for r in rows})
        if signatures['fast_v1_serial']!=signatures['fast_v2_parallel']:
            raise RuntimeError('parallel SIX/probability/motion/integer counts byte gate failed')
        for repeat in range(repeats):
            for name in (tuple(arms) if repeat%2==0 else tuple(reversed(arms))):
                pool=pools[name]; pool.reset(indices,chunk); tick=time.perf_counter()
                rows=[r for batch in pool.batches(indices,chunk,stop_event=stop_event) for r in batch]
                if len(rows)!=len(indices): raise InterruptedError('parallel speed stopped')
                durations[name].append((time.perf_counter()-tick)/len(indices))
        mean={k:float(np.mean(v)) for k,v in durations.items()}
        return dict(windows=len(indices),repeats=repeats,processes=processes,chunk=chunk,
            seconds_per_window=mean,speedup=mean['fast_v1_serial']/mean['fast_v2_parallel'],
            repeat_seconds_per_window=dict(durations),startup_and_warmup_seconds=startup,
            worker_pids=worker_pids,counts_exact=True,probability_and_six_dense_bytes_exact=True,
            scope='same-window actual eval incl raw geometry/SIX dense/GT/metrics/IPC; NOT formal FPS',
            timing_policy='alternating arms, reset frame geometry each pass, warm ONLY first window/worker',
            worker_stage_seconds_are_overlapping=True,no_metric_cursor_updates=True)
    finally:
        for pool in pools.values(): pool.close()
