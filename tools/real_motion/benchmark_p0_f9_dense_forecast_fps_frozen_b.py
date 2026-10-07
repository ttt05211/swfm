#!/usr/bin/env python3
"""Formal Dense Forecast FPS benchmark for the frozen Point-CCR B rule.

Frozen B:
  weighted epoch-boundary Point-CCR checkpoint;
  raw ADD sigmoid @ 0.5;
  REMOVE disabled.

The timing boundary is IDENTICAL to p0_f9_dense_forecast_fps_final_v1:
prepared CausalHistoryState -> live KTA/Strong -> V18 motion/transport ->
Point CCR -> constrained six dense semantic occupancy frames.

Exactly 20 frozen DEV64 windows x 3 repeats are timed. History representation,
disk I/O, GT/metrics, checkpoint load, compile and warm-up remain excluded.
The benchmark additionally reports P90 and CUDA peak allocated/reserved memory.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None,""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from real_motion.ccr_frozen_b import frozen_b_probabilities, effective_corrected_add_thresholds
from real_motion.column_runtime_pipeline import CachedColumnSource
from real_motion.final_dataflow import prepare_history, forecast_six, build_causal_motion_prior
from real_motion.native_column_cpu import prepare_native, get_prepared_native
from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.runtime_config import add_config_args, load_runtime_config, make_prepare_config
from real_motion.v21_source_induction import stable_json_fingerprint
from tools.real_motion.eval_p0_f9_v18_se2 import load_cache
from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import load_manifest, sha256
from tools.real_motion.joint_training_recovery import snapshot_checkpoint
from tools.real_motion.pilot_p0_f9_canonical_causal_repair import probabilities as corrected_probabilities
from tools.real_motion.point_ccr_v18_fps_common import load_point_head, select_population
from tools.real_motion.run_p0_f9_shared_evidence_pilot import PilotProvider, require_cuda
from tools.real_motion.static_evidence_selector_common import CLEAN_SHA256, write_json, finite_json
from tools.real_motion.train_p0_f9_causal_columns import record_keys
from tools.real_motion.train_p0_f9_joint_causal_columns import load_joint


PROTOCOL="p0_f9_dense_forecast_fps_frozen_b_v1"
WINDOWS=20
STRESS_WINDOWS=2
REPEATS=3
FUTURE_FRAMES=6


def _hash_dense(dense):
    import hashlib
    return [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in dense]


def _aggregate(trials):
    seconds=np.asarray([x["seconds"] for x in trials],np.float64)
    peak=np.asarray([x["peak_allocated_mib"] for x in trials],np.float64)
    inc=np.asarray([x["incremental_peak_allocated_mib"] for x in trials],np.float64)
    reserved=np.asarray([x["peak_reserved_mib"] for x in trials],np.float64)
    if len(seconds)!=WINDOWS*REPEATS or not np.isfinite(seconds).all() or np.any(seconds<=0):
        raise RuntimeError("incomplete/invalid frozen-B FPS population")
    total=float(seconds.sum())
    stage_keys=sorted(set().union(*(x["host_stages_seconds"] for x in trials)))
    return dict(
        samples=len(seconds),windows=len({x["key"] for x in trials}),
        future_frames=FUTURE_FRAMES*len(seconds),
        total_seconds=total,dense_forecast_fps=FUTURE_FRAMES*len(seconds)/total,
        mean_six_ms=1000*float(seconds.mean()),
        p50_six_ms=1000*float(np.median(seconds)),
        p90_six_ms=1000*float(np.percentile(seconds,90)),
        peak_allocated_mib_max=float(peak.max()),
        peak_allocated_mib_p90=float(np.percentile(peak,90)),
        incremental_peak_allocated_mib_max=float(inc.max()),
        peak_reserved_mib_max=float(reserved.max()),
        host_stage_mean_ms={
            k:1000*float(np.mean([x["host_stages_seconds"].get(k,0.) for x in trials]))
            for k in stage_keys},
        formula="total_future_frames / total_synchronized_wall_time")


def _summary(r):
    lines=["===== FROZEN B FORMAL DENSE FORECAST FPS =====","status="+r["status"],
           "protocol="+PROTOCOL,
           "rule=raw weighted ADD sigmoid@0.5; REMOVE off",
           "boundary=CausalHistoryState -> six finished dense semantic occupancy frames",
           "population=20 frozen windows x 3 repeats; batch=1",
           "EXCLUDED: history representation, disk I/O, GT/metrics, checkpoint load, compile/warm-up"]
    if r.get("decision_rule"):
        lines.append("decision="+json.dumps(r["decision_rule"],sort_keys=True))
    if r.get("aggregate"):
        a=r["aggregate"]
        lines += [
            f'Dense Forecast FPS={a["dense_forecast_fps"]:.3f}',
            f'six-frame latency mean={a["mean_six_ms"]:.3f} ms p50={a["p50_six_ms"]:.3f} ms p90={a["p90_six_ms"]:.3f} ms',
            f'GPU peak allocated max={a["peak_allocated_mib_max"]:.1f} MiB incremental max={a["incremental_peak_allocated_mib_max"]:.1f} MiB reserved max={a["peak_reserved_mib_max"]:.1f} MiB',
            "host stage means="+json.dumps(a["host_stage_mean_ms"],sort_keys=True)]
    if r.get("parity"):
        lines.append("raw-vs-corrected-equivalent dense parity="+json.dumps(r["parity"],sort_keys=True))
    lines.append("No persistent VAL geometry cache is used inside or outside the official forecast timing boundary.")
    if "error" in r:lines.append("error="+r["error"])
    return "\n".join(lines)+"\n"


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);add_config_args(p)
    for key in ("checkpoint","ccr-checkpoint","base-checkpoint","dev-cache",
                "population-manifest","dataroot","dev-info","out-dir"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--cpu-workers",type=int,default=10)
    p.add_argument("--ccr-cpu-workers",type=int,default=4)
    a=p.parse_args(argv)
    out=Path(a.out_dir)
    if out.exists():p.error("fresh output required")
    for key in ("config","checkpoint","ccr_checkpoint","base_checkpoint","dev_cache",
                "population_manifest","dev_info"):
        if not Path(getattr(a,key) or "").is_file():p.error("missing "+key)
    if not Path(a.dataroot).is_dir() or not 1<=a.cpu_workers<=16 or not 1<=a.ccr_cpu_workers<=8:
        p.error("invalid paths/workers")

    device=require_cuda(a.device);torch.set_num_threads(1)
    out.mkdir(parents=True);started=time.perf_counter()
    result=dict(
        status="running",protocol=PROTOCOL,GPU=torch.cuda.get_device_name(device),
        torch_version=str(torch.__version__),windows=WINDOWS,repeats=REPEATS,
        official_boundary="CausalHistoryState_to_six_dense_semantic_occupancy",
        no_training=True,persistent_val_cache_used=False,trials=[],
        parity=dict(checked_windows=0,passed_windows=0))
    def persist():
        write_json(out/"dense_forecast_fps_frozen_b.json",result)
        (out/"summary.txt").write_text(_summary(result),encoding="utf-8")
    persist()

    pool=ThreadPoolExecutor(max_workers=a.ccr_cpu_workers)
    try:
        tick=time.perf_counter()
        result["native_preflight"]=prepare_native(out/"native_build")
        result["compile_seconds_excluded"]=time.perf_counter()-tick
        kernels=get_prepared_native()

        sources={"epoch19":a.checkpoint,"point_ccr":a.ccr_checkpoint,"clean_e14":a.base_checkpoint}
        snapshots={k:out/(k+"_snapshot.pt") for k in sources}
        digests={k:snapshot_checkpoint(v,snapshots[k]) for k,v in sources.items()}
        result["checkpoint_sha256"]=digests

        cfg=load_runtime_config(a.config,a.override);config_fp=stable_json_fingerprint(cfg)
        ck,teacher=load_joint(
            snapshots["epoch19"],device,reference_sha=CLEAN_SHA256,
            config_sha=config_fp,allow_diagnostic=True)
        if (teacher.transport.config.history_frames!=4 or ck.get("cursor_epoch")!=19
                or ck["model_configs"].get("adaptive_context") is not None):
            raise RuntimeError("selected strict-four epoch19 transport required")
        teacher.eval().requires_grad_(False)
        for path,expected in ((a.dev_cache,ck["cache_fingerprints"]["dev"]),
                              (a.dev_info,ck["info_fingerprints"]["dev"]),
                              (snapshots["clean_e14"],CLEAN_SHA256)):
            if sha256(path)!=expected:raise RuntimeError("checkpoint/data provenance mismatch: "+str(path))

        saved=torch.load(snapshots["point_ccr"],map_location="cpu",weights_only=False)
        head=load_point_head(
            saved,teacher_sha256=digests["epoch19"],config_fingerprint=config_fp,
            source_dim=teacher.columns.source_dim,device=device,
            allow_completed_epoch_boundary=True)
        result["decision_rule"]={
            "name":"frozen_B","ADD_score":"sigmoid(raw_weighted_logit)",
            "ADD_threshold":.5,"REMOVE":False,
            **effective_corrected_add_thresholds(head)}
        result["point_checkpoint_epoch"]=int(saved.get("epoch",0))
        result["point_checkpoint_updates"]=int(saved.get("updates",0))

        manifest,keys64,_=load_manifest(a.population_manifest)
        parent=tuple(map(tuple,manifest["parent_keys"]))
        if (len(keys64)!=64 or len(parent)!=512
                or manifest["manifest_fingerprint"]!=ck["dev_manifest_fingerprint"]
                or tuple(map(tuple,ck["dev_keys"]))!=parent
                or saved["contract"]["dev_manifest_fingerprint"]!=manifest["manifest_fingerprint"]):
            raise RuntimeError("frozen FPS population identity mismatch")
        _,records=load_cache(a.dev_cache);record_keys(records)
        chosen,population=select_population(
            records,keys64,windows=WINDOWS,stress_windows=STRESS_WINDOWS)
        del records
        result["population"]=dict(
            windows=len(chosen),scenes=len({r["scene_name"] for r in chosen}),
            manifest_fingerprint=manifest["manifest_fingerprint"],
            key_fingerprint=stable_json_fingerprint([m["key"] for m in population]),
            selection="same scene-balanced + source-count stress rule as official final FPS")
        write_json(out/"fps_manifest.json",{
            "protocol":PROTOCOL,**result["population"],"keys":population})

        provider=PilotProvider(
            snapshots["clean_e14"],CLEAN_SHA256,make_prepare_config(cfg),
            device,a.cpu_workers,teacher,None)
        source=CachedColumnSource(
            NuScenesWindowSource(a.dataroot,info_pkl=a.dev_info,verbose=False),128)
        source.copy_on_insert=False

        rule=effective_corrected_add_thresholds(head)
        def corrected_equivalent(head0,evidence,plan,output,device0):
            p0=corrected_probabilities(head0,evidence,plan,output,device0)
            cut=np.where(
                evidence.actor[:,None]>=0,
                rule["dynamic_corrected_threshold"],rule["static_corrected_threshold"])
            outp=np.zeros_like(p0,np.float32)
            outp[...,0]=(p0[...,0]>=cut).astype(np.float32)
            return outp

        with (out/"progress.jsonl").open("x",encoding="utf-8") as log:
            for wi,(record,meta) in enumerate(zip(chosen,population),1):
                history=prepare_history(provider,source,record,kernels=kernels,executor=pool)
                live_kta,live_anchors=build_causal_motion_prior(history,provider.pcfg.frame_dt_s)
                if not np.allclose(
                        live_kta,torch.as_tensor(record["kta_displacement_xy_m"]).cpu().numpy(),
                        rtol=0,atol=1e-6):
                    raise RuntimeError("live KTA/cache mismatch")
                if not np.allclose(
                        live_anchors,torch.as_tensor(record["anchors_xy_t0_m"]).cpu().numpy(),
                        rtol=0,atol=1e-5):
                    raise RuntimeError("live KTA anchors/cache mismatch")

                # Warm-up outside timing.
                forecast_six(
                    history,provider,teacher.transport,head,frozen_b_probabilities,
                    kernels=kernels,executor=pool,majority_backend="native")
                torch.cuda.synchronize(device)

                # Real-data parity outside timing: raw ADD@.5 must equal the
                # algebraically equivalent role-specific corrected decision.
                raw_ref=forecast_six(
                    history,provider,teacher.transport,head,frozen_b_probabilities,
                    kernels=kernels,executor=pool,majority_backend="native")
                eq_ref=forecast_six(
                    history,provider,teacher.transport,head,corrected_equivalent,
                    kernels=kernels,executor=pool,majority_backend="native")
                torch.cuda.synchronize(device)
                result["parity"]["checked_windows"]+=1
                if _hash_dense(raw_ref["dense"])!=_hash_dense(eq_ref["dense"]):
                    raise RuntimeError("frozen-B raw/corrected-equivalent dense parity mismatch")
                result["parity"]["passed_windows"]+=1
                del raw_ref,eq_ref

                for repeat in range(REPEATS):
                    torch.cuda.synchronize(device)
                    start_alloc=torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    tick=time.perf_counter()
                    output=forecast_six(
                        history,provider,teacher.transport,head,frozen_b_probabilities,
                        kernels=kernels,executor=pool,majority_backend="native")
                    torch.cuda.synchronize(device)
                    elapsed=time.perf_counter()-tick
                    peak_alloc=torch.cuda.max_memory_allocated(device)
                    peak_reserved=torch.cuda.max_memory_reserved(device)
                    if not math.isfinite(elapsed) or elapsed<=0:
                        raise RuntimeError("invalid synchronized wall time")
                    row=dict(
                        key="/".join(meta["key"]),stratum=meta["stratum"],
                        sources=meta["sources"],repeat=repeat+1,seconds=elapsed,
                        host_stages_seconds=output["stages_seconds"],
                        six_complete_dense=output["six_complete_dense"],
                        start_allocated_mib=start_alloc/2**20,
                        peak_allocated_mib=peak_alloc/2**20,
                        incremental_peak_allocated_mib=max(0,peak_alloc-start_alloc)/2**20,
                        peak_reserved_mib=peak_reserved/2**20)
                    result["trials"].append(row)
                    log.write(json.dumps(finite_json(row),allow_nan=False)+"\n");log.flush()

                result["aggregate"]=_aggregate(result["trials"])
                persist()
                print(
                    f'FROZEN_B_FPS {wi}/{WINDOWS} FPS={result["aggregate"]["dense_forecast_fps"]:.3f} '
                    f'p90={result["aggregate"]["p90_six_ms"]:.2f}ms '
                    f'peak={result["aggregate"]["peak_allocated_mib_max"]:.1f}MiB',
                    flush=True)
                del history

        if result["parity"]["passed_windows"]!=WINDOWS:
            raise RuntimeError("incomplete frozen-B parity gate")
        result.update(status="complete",elapsed_seconds=time.perf_counter()-started)
        persist();print(_summary(result),flush=True);return 0
    except BaseException as exc:
        result.update(status="failed",error=type(exc).__name__+": "+str(exc),
                      elapsed_seconds=time.perf_counter()-started)
        persist();raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        persist()


if __name__=="__main__":
    raise SystemExit(main())
