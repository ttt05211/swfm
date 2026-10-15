"""Frozen real subset research. No future occupancy labels enter forecasting.

Historical GT-feature oracle and supervised adapters are explicitly separated
from valid STC controls. This evaluator itself never runs an optimizer.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_evaluation.predictor import Predictor
from tools.shared_source_track.adapter import SharedTrackPredictor
from .bundle import Bundle
from .replay import VIEWS, prepare_forecast, iter_forecasts, temporal_static_prior, multiframe_velocity


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument('--bundle', default='../stc_overnight_bundle.zip')
    a.add_argument('--out', required=True)
    a.add_argument('--model', default='dense_epoch2')
    a.add_argument('--view', choices=list(VIEWS), default='identity')
    a.add_argument('--modality', choices=['stc', 'gt'], default='stc')
    a.add_argument('--known', choices=['dense_valid', 'lidar_known'], default='dense_valid')
    a.add_argument('--device', default='cuda')
    a.add_argument('--limit', type=int, default=64)
    a.add_argument('--save-logits', action='store_true')
    a.add_argument('--group-size', type=int, default=1)
    a.add_argument('--temporal-static', action='store_true')
    a.add_argument('--occupied-prior-half', action='store_true')
    a.add_argument('--feature-affine')
    a.add_argument('--oracle-refiner-history', action='store_true')
    a.add_argument('--encoder-checkpoint')
    a.add_argument('--history-adapter')
    a.add_argument('--held-split',help='adapter contract, to evaluate ONLY its held scenes even for original baseline')
    a.add_argument('--memory-recency',action='store_true')
    a.add_argument('--kta-multiframe',action='store_true')
    a.add_argument('--memory-current-only',action='store_true')
    a.add_argument('--importance-uniform',action='store_true')
    a.add_argument('--shared-tracks-for-baseline',action='store_true')
    a.add_argument('--center-filter',action='store_true')
    a.add_argument('--temporal-support',action='store_true')
    a.add_argument('--memory-static-current',action='store_true')
    a.add_argument('--memory-range',action='store_true')
    a.add_argument('--velocity-model',choices=['half','full','wiener'],default='half')
    a.add_argument('--role',choices=['reference64','clean_train'],default='reference64')
    a.add_argument('--posterior-stats',action='store_true')
    a.add_argument('--spatial-stats',action='store_true')
    a.add_argument('--memory-last2',action='store_true')
    a.add_argument('--posterior-bias')
    a.add_argument('--temporal-strict-fill',action='store_true')
    a.add_argument('--occupancy-head')
    args = a.parse_args(argv)
    torch.set_num_threads(1)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    b = Bundle(args.bundle)
    rows = [r for r in b.records if r['role'] == args.role][:args.limit]
    if args.held_split:
        split=json.loads(Path(args.held_split).read_text(encoding='utf-8'))
        held=set(map(tuple,split['held_keys']));rows=[r for r in rows if (r['scene_name'],r['t0_token']) in held]
    model, saved, policy = b.load_model(args.model, args.device)
    occupancy_head=None;occupancy_saved=None
    if args.occupancy_head:
        from .occupancy_head import OccupancyCalibration
        occupancy_saved=torch.load(args.occupancy_head,map_location='cpu',weights_only=False)
        if occupancy_saved['contract']['source_sha256']!=saved['source_sha256']:raise ValueError('head belongs to another body')
        occupancy_head=OccupancyCalibration().to(args.device).eval().requires_grad_(False)
        occupancy_head.load_state_dict(occupancy_saved['model'],strict=True)
    original_policy=policy
    if args.shared_tracks_for_baseline:policy='shared_pinned'
    if args.importance_uniform:
        model.columns.importance.register_forward_hook(lambda module,inputs,output:torch.zeros_like(output))
    history_adapter = None; adapter_saved = None
    if args.history_adapter:
        from .adapter import FeatureAdapter
        adapter_saved=torch.load(args.history_adapter,map_location='cpu',weights_only=False)
        history_adapter=FeatureAdapter().to(args.device).eval().requires_grad_(False)
        history_adapter.load_state_dict(adapter_saved['model'],strict=True)
        fit=set(adapter_saved['contract']['fit_scenes'])
        if any(r['scene_name'] in fit for r in rows):raise RuntimeError('adaptation fit scene supplied to evaluation')
    adapted = None
    if args.encoder_checkpoint:
        adapted = torch.load(args.encoder_checkpoint,map_location='cpu',weights_only=False)
        model.transport.history_encoder.load_state_dict(adapted['model'],strict=True)
    affine = None
    if args.feature_affine:
        affine = json.loads(Path(args.feature_affine).read_text(encoding='utf-8'))
        scale = torch.tensor(affine['scale'],device=args.device)[None,:,None,None]
        bias = torch.tensor(affine['bias'],device=args.device)[None,:,None,None]
        model.transport.history_encoder.register_forward_hook(lambda module,inputs,value:
            (value.float()*scale+bias).to(value.dtype))
    predictor = (SharedTrackPredictor if policy == 'shared_pinned' else Predictor)(model, b.pcfg, args.device, workers=2)
    contract = dict(bundle_manifest_sha256=hashlib.sha256(b.zip.read('manifest.json')).hexdigest(),
        model_sha256=next(v['sha256'] for v in b.manifest['weights'] if v['role'] == args.model),
        model=args.model, source_policy=policy, modality=args.modality, known=args.known, view=args.view,
        device=args.device, group_size=args.group_size, scientific_updates=0,
        keys=[[r['scene_name'], r['t0_token']] for r in rows],
        full_grid=True, future_GT_ego_conditioned=True, moving_not_evaluated=True,
        scope='repeatedly inspected reference64 development subset; NOT full VAL target verification')
    if args.temporal_static: contract['temporal_static'] = 'strict_three_history_votes_static_only_v1'
    if args.occupied_prior_half: contract['occupied_prior_half'] = 'final_c05_on_prior_occupied_only_free_unchanged_v1'
    if affine is not None:contract['feature_affine'] = affine
    if adapted is not None:contract['adapted_encoder'] = adapted['contract']
    if adapter_saved is not None:contract['supervised_HISTORY_adapter'] = adapter_saved['contract']
    if args.memory_recency:contract['memory_recency'] = 'fixed_half_second_decay_0.125_0.25_0.5_1'
    if args.kta_multiframe:contract['kta_multiframe'] = 'EXPERIMENT_half_original_pair_plus_multiframe_LS_no_new_match'
    if args.memory_current_only:contract['memory_current_only'] = 'DIAGNOSTIC_only_t0_memory_STWM_still_four_frames'
    if args.importance_uniform:contract['importance_uniform'] = 'DIAGNOSTIC_disable_learned_evidence_scores_static_and_dynamic'
    if args.shared_tracks_for_baseline:contract['original_source_policy']=original_policy
    if args.center_filter:contract['center_filter']='half_observed_t0_plus_linear_fit_intercept_GEOMETRY_AND_MEMORY'
    if args.temporal_support:contract['temporal_support']='soft_current_free_with_two_past_static_agreements_half_probability_v1'
    if args.memory_static_current:contract['memory_static_current']='static_t0_only_dynamic_all4'
    if args.memory_range:contract['memory_range']='local_range_reliability_1_over_1_plus_r2_over_20m2'
    if args.velocity_model!='half':contract['velocity_model']=args.velocity_model
    if args.role!='reference64':contract['role']=args.role
    if args.posterior_stats:contract['posterior_stats']=True
    if args.spatial_stats:contract['spatial_stats']='binary_by_t0_range_and_future_z'
    if occupancy_saved is not None:contract['occupancy_head']={k:v for k,v in occupancy_saved.items() if k!='model'}
    if args.memory_last2:contract['memory_last2']='memory_latest_two_frames_STWM_four_frames'
    if args.temporal_strict_fill:contract['temporal_strict_fill']='past3_same_static_class_fills_current_free_only'
    posterior_bias=None
    if args.posterior_bias:
        fitted=json.loads(Path(args.posterior_bias).read_text(encoding='utf-8'))
        if fitted['target_contract']['model_sha256']!=contract['model_sha256']:raise ValueError('bias fit to other model')
        posterior_bias=torch.tensor(fitted['biases']['binary_relative_alignment'],device=args.device,dtype=torch.float32)[None,:,None]
        contract['posterior_bias_fit']=fitted
    if args.oracle_refiner_history:
        contract['oracle_refiner_history'] = 'GT historical feature oracle ONLY; NOT a valid STC score'
    path = out/'contract.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != contract:
        raise RuntimeError('output contract differs; choose new directory')
    dump(path, contract)
    counts = {n: metrics.empty(3) for n in ('transport', 'semantic_refine')}
    view = VIEWS[args.view]; report_hs = (1, 3, 5); started = time.monotonic()
    try:
        for i, r in enumerate(rows):
            result_file = out/f'window_{i:03d}.json'
            if result_file.exists():
                result = json.loads(result_file.read_text(encoding='utf-8'))
                if result['key'] != contract['keys'][i]: raise RuntimeError('window key differs')
                for n in counts: counts[n]['confusion'] += np.asarray(result['confusion'][n], np.int64)
                continue
            rec, raw, times = b.history(r, args.modality, args.known)
            rec, raw, poses = view.apply(rec, raw, r['future_poses'], b.pcfg.grid)
            row = predictor.prepare(rec, raw, times=times)
            kta_audit=None
            if args.kta_multiframe or args.center_filter:row,kta_audit=multiframe_velocity(row,predictor.provider,times,args.center_filter,args.velocity_model)
            if args.memory_recency or args.memory_current_only or args.memory_last2:
                # Geometry and network features remain original; only normalized
                # history evidence pooling receives a fixed age preference.
                reads=row[3][0]['memory_reads'];weights=np.asarray([.125,.25,.5,1.],np.float32)
                if args.memory_current_only:weights=np.asarray([0.,0.,0.,1.],np.float32)
                if args.memory_last2:weights=np.asarray([0.,0.,1.,1.],np.float32)
                reads['static_valid']=reads['static_valid']*weights[:,None,None,None]
                reads['coverage']=reads['coverage']*weights[reads['frame']]
            if args.memory_static_current:
                reads=row[3][0]['memory_reads']
                reads['static_valid']=reads['static_valid']*np.asarray([0.,0.,0.,1.],np.float32)[:,None,None,None]
            if args.memory_range:
                grid=b.pcfg.grid;nx,ny,nz=grid.shape_hwd;dx,dy,_=grid.voxel_size
                xx=grid.x_min+(np.arange(nx//2)+.5)*2*dx
                yy=grid.y_min+(np.arange(ny//2)+.5)*2*dy
                reliability=(1/(1+(xx[:,None]**2+yy[None,:]**2)/400)).astype(np.float32)
                reads=row[3][0]['memory_reads'];reads['static_valid']=reads['static_valid']*reliability[None,:,:,None]
                cells=reads['cell'];reads['coverage']=reads['coverage']*reliability[cells//((ny//2)*nz),(cells//nz)%(ny//2)]
            context = prepare_forecast(predictor, row, poses)
            if history_adapter is not None:
                from tools.shared_history_experiments.model import encode_window
                with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    field,_,_=encode_window(model,row[1],row[3],row[2][0],predictor.provider.device)
                    context['lifted']=model.columns.lift_history(history_adapter(field))
                del field
            if args.oracle_refiner_history:
                _, clean, _ = b.history(r,'gt','dense_valid')
                with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    f = model.transport.history_encoder(torch.as_tensor(clean['history_occ'],device=args.device),
                        torch.as_tensor(clean['history_observed'],device=args.device))
                    context['lifted'] = model.columns.lift_history(f)
                del clean, f
            static_audit = temporal_static_prior(context,support_only=args.temporal_support,strict_fill=args.temporal_strict_fill) if (args.temporal_static or args.temporal_support or args.temporal_strict_fill) else None
            predictions = {n: [] for n in counts}; logits_saved = []
            posterior_sum=[]
            spatial_sum=[];spatial_counts=[]
            for hs, fw in iter_forecasts(context, group_size=args.group_size):
                if occupancy_head is not None:
                    from .occupancy_head import features,adjust
                    with torch.no_grad():fw['logits']=adjust(fw['logits'],occupancy_head(features(fw['memory'],fw['prior'],fw['shape']),fw['shape']))
                if posterior_bias is not None:fw['logits']=fw['logits']+posterior_bias
                if args.occupied_prior_half:
                    occupied = fw['prior'].argmax(1) != 17
                    base = model.columns.prior_logits(fw['prior'])
                    fw['logits'] = fw['logits']-.5*base*occupied[:,None]
                for j, h in enumerate(hs):
                    if h not in report_hs: continue
                    if args.posterior_stats:posterior_sum.append(fw['logits'][j].softmax(0).double().sum(-1).cpu().numpy())
                    if args.spatial_stats:
                        from .range_calibration import bins,posterior_stats
                        ids=bins(raw,poses,b.pcfg.grid,h,args.device)
                        sm,nn=posterior_stats(fw['logits'][j],ids,5*b.pcfg.grid.shape_hwd[-1])
                        spatial_sum.append(sm);spatial_counts.append(nn)
                    for n, k in (('transport', 'prior'), ('semantic_refine', 'logits')):
                        val = view.restore(fw[k][j], fw['shape'])
                        predictions[n].append(val.argmax(0).byte().cpu().numpy())
                        if args.save_logits and n == 'semantic_refine':
                            logits_saved.append(val.float().cpu().numpy())
                del fw
            # Future occupancy labels are first read after ALL SIX forecast readouts.
            target = b.targets(r)
            local = {n: metrics.empty(3) for n in counts}
            for n in counts:
                for h in range(3): metrics.add(local[n], h, predictions[n][h], target[h])
                counts[n]['confusion'] += local[n]['confusion']
            if args.save_logits:
                # Local scratch is intentionally uncompressed: float32 is exact and
                # repeated DEFLATE was taking much longer than the GPU forecast.
                extra = dict(valid=view.valid(context['shape'])) if hasattr(view,'valid') else {}
                np.savez(out/f'logits_{i:03d}.npz', logits=np.stack(logits_saved), **extra)
            result = dict(key=contract['keys'][i], confusion={n: local[n]['confusion'].tolist() for n in local},
                          static_audit=static_audit)
            result['kta_audit']=kta_audit
            if args.posterior_stats:result['posterior_sum']=np.stack(posterior_sum).tolist()
            if args.spatial_stats:
                result['spatial_occ_sum']=np.stack(spatial_sum).tolist();result['spatial_counts']=np.stack(spatial_counts).tolist()
            dump(result_file, result)
            current = {n: metrics.report(v, (1., 2., 3.), moving=False) for n, v in counts.items()}
            dump(out/'results.json', dict(status='running', windows=i+1, contract=contract, metrics=current))
            print('LOCAL_REAL', args.model, args.view, i+1, '/', len(rows),
                'seconds=', round(time.monotonic()-started, 2),
                'metrics=', {n: v['average'] for n, v in current.items()}, flush=True)
            del row, context, raw, predictions, logits_saved, target, val
            gc.collect()
        current = {n: metrics.report(v, (1., 2., 3.), moving=False) for n, v in counts.items()}
        dump(out/'results.json', dict(status='complete', windows=len(rows), contract=contract, metrics=current,
                                     seconds=time.monotonic()-started))
    finally: predictor.close(); b.close()


if __name__ == '__main__': main()
