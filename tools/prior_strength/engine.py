import copy
import time
import numpy as np
import torch
from tools.final_semantic_evaluation import metrics
from tools.final_semantic_diagnostics.input_audit import identity

INDICES = (1, 3, 5)
HORIZONS = (1., 2., 3.)
INPUTS = ('occ_gt', 'stc_gt')
READOUTS = ('transport', 'original_c1', 'diagnostic_c05')
GROUPS = ('all', 'wrong_free', 'false_occupancy', 'wrong_class')
EDIT_FIELDS = ('count', 'changed', 'corrected', 'damaged', 'wrong_to_other_wrong')


def reweight(logits, prior_logits, strength):
    """Change ONLY the additive final log-prior; retain the original residual.

    Reuse the original forward, including its unmodified coarse/point priors.
    c=1 is a no-op, not a subtraction/readdition that could round differently.
    """
    if strength not in (1., .5): raise ValueError('only preregistered strengths 1 and 0.5 allowed')
    if logits.shape != prior_logits.shape or logits.ndim != 3 or logits.shape[1] != 18:
        raise ValueError('full semantic logits/prior shape mismatch')
    if not torch.isfinite(logits).all() or not torch.isfinite(prior_logits).all():
        raise ValueError('nonfinite frozen readout')
    return logits if strength == 1. else logits + (strength-1.)*prior_logits


def edit_counts(before, after, target):
    """Semantic correctness, not binary occupancy correctness or mIoU gain."""
    before, after, target = (np.asarray(v) for v in (before, after, target))
    if before.shape != after.shape or before.shape != target.shape:
        raise ValueError('edit grid mismatch')
    if any(v.dtype.kind not in 'iu' or (v < 0).any() or (v > 17).any() for v in (before, after, target)):
        raise ValueError('edit labels must be integer 0..17')
    wrong, final_wrong, changed = before != target, after != target, before != after
    groups = dict(all=np.ones(before.shape, bool),
        wrong_free=(before == 17) & (target != 17),
        false_occupancy=(before != 17) & (target == 17),
        wrong_class=(before != 17) & (target != 17) & wrong)
    return {name: dict(count=int(mask.sum()), changed=int((mask & changed).sum()),
        corrected=int((mask & wrong & ~final_wrong).sum()),
        damaged=int((mask & ~wrong & final_wrong).sum()),
        wrong_to_other_wrong=int((mask & wrong & final_wrong & changed).sum()))
        for name, mask in groups.items()}


@torch.no_grad()
def row_report(predictor, source, job, artifact=None):
    identity(source, job)
    predictions = {}
    reference_record, reference_poses, reference_times = None, None, None
    poses = source.poses(job)  # explicit GT trajectory conditioning, NOT occupancy
    for name, kind in zip(INPUTS, ('occ', 'stc')):
        record, raw, times = source.history(job, kind)
        if reference_record is None:
            reference_record, reference_poses, reference_times = record, np.asarray(raw['history_poses']), times
        elif (record != reference_record or not np.array_equal(raw['history_poses'], reference_poses)
                or not np.array_equal(times, reference_times)):
            raise RuntimeError('GT/STC historical identities/poses/times differ')
        if kind == 'stc' and artifact:
            for token, sem in zip(job.history_tokens, raw['history_occ']): artifact.check(job.scene_name, token, sem)
        fw = predictor.forward(predictor.prepare(record, raw, times=times), poses=poses)
        if fw['prior'].shape[0] != 6 or fw['logits'].shape != fw['prior'].shape:
            raise ValueError('all six future semantic readouts required')
        ids = torch.tensor(INDICES, device=fw['logits'].device)
        probability = fw['prior'].index_select(0, ids)
        logits = fw['logits'].index_select(0, ids)
        p = predictor.body.columns.prior_logits(probability)
        shape = tuple(fw['shape'])
        predictions[name] = dict(transport=probability.argmax(1).reshape(3, *shape).byte().cpu().numpy())
        for label, strength in (('original_c1', 1.), ('diagnostic_c05', .5)):
            predictions[name][label] = reweight(logits, p, strength).argmax(1).reshape(3, *shape).byte().cpu().numpy()
        del fw, probability, logits, p
    # Complete both modalities/readouts before any future occupancy or annotation read.
    targets = source.targets(job, INDICES)
    moving = source.moving(job, INDICES, HORIZONS, predictor.provider.pcfg.grid, predictor.provider.workers)
    counts, edits = {}, {}
    for name in INPUTS:
        out = predictions[name]
        for label in READOUTS:
            c = metrics.empty(3)
            for h in range(3): metrics.add(c, h, out[label][h], targets[h], moving[h])
            counts[name+'/'+label] = {k: v.tolist() for k, v in c.items()}
        edits[name] = {label: [edit_counts(out['transport'][h], out[label][h], targets[h]) for h in range(3)]
                      for label in READOUTS[1:]}
        edits[name]['half_vs_original'] = [edit_counts(out['original_c1'][h], out['diagnostic_c05'][h], targets[h])
                                           for h in range(3)]
    return dict(key=[job.scene_name, job.t0_token], counts=counts, edits=edits)


def checked_rows(state, keys, contract, volume):
    if state.get('contract') != contract or not isinstance(state.get('rows'), list) or len(state['rows']) > len(keys):
        raise RuntimeError('diagnostic contract/cursor changed')
    expected = {n+'/'+r for n in INPUTS for r in READOUTS}
    for i, row in enumerate(state['rows']):
        if row['key'] != keys[i]: raise RuntimeError('diagnostic population prefix changed')
        if set(row['counts']) != expected: raise RuntimeError('saved routes differ')
        for value in row['counts'].values():
            if set(value) != {'confusion', 'moving'}: raise RuntimeError('saved metric fields differ')
            c, m = (np.asarray(value[k]) for k in ('confusion', 'moving'))
            if (c.shape != (3, 18, 18) or m.shape != c.shape or c.dtype.kind not in 'iu' or m.dtype.kind not in 'iu'
                    or (c < 0).any() or (m < 0).any() or (m > c).any() or np.any(c.sum((1, 2)) != volume)):
                raise RuntimeError('saved integer metrics invalid')
        if set(row['edits']) != set(INPUTS): raise RuntimeError('saved edit inputs differ')
        for input_edits in row['edits'].values():
            if set(input_edits) != {'original_c1', 'diagnostic_c05', 'half_vs_original'}:
                raise RuntimeError('saved edit comparisons differ')
            for horizons in input_edits.values():
                if len(horizons) != 3: raise RuntimeError('saved edit horizons differ')
                for groups in horizons:
                    if set(groups) != set(GROUPS): raise RuntimeError('saved edit groups differ')
                    for item in groups.values():
                        if (set(item) != set(EDIT_FIELDS) or any(type(v) is not int or v < 0 or v > volume for v in item.values())
                                or item['changed'] > item['count']
                                or item['corrected']+item['damaged']+item['wrong_to_other_wrong'] != item['changed']):
                            raise RuntimeError('saved edit counts invalid')
                    if groups['all']['count'] != volume: raise RuntimeError('saved edit population differs')
        if not np.isfinite(row['seconds']) or row['seconds'] < 0: raise RuntimeError('invalid elapsed time')


def aggregate(rows):
    reports = {}
    for name in rows[0]['counts']:
        counts = {k: np.sum([np.asarray(r['counts'][name][k], np.int64) for r in rows], axis=0)
                  for k in ('confusion', 'moving')}
        reports[name] = metrics.report(counts, HORIZONS)
    edits = {}
    for name in INPUTS:
        edits[name] = {}
        for comparison in ('original_c1', 'diagnostic_c05', 'half_vs_original'):
            horizons = []
            for h in range(3):
                groups = {group: {key: sum(r['edits'][name][comparison][h][group][key] for r in rows)
                          for key in EDIT_FIELDS} for group in GROUPS}
                for item in groups.values(): item['net_corrected'] = item['corrected']-item['damaged']
                horizons.append(groups)
            edits[name][comparison] = dict(per_horizon= {str(t): v for t, v in zip(HORIZONS, horizons)},
                totals={g: {k: sum(v[g][k] for v in horizons) for k in (*EDIT_FIELDS, 'net_corrected')} for g in GROUPS})
    delta = {name: {k: reports[name+'/diagnostic_c05']['average'][k]-reports[name+'/original_c1']['average'][k]
             if reports[name+'/diagnostic_c05']['average'][k] is not None and reports[name+'/original_c1']['average'][k] is not None else None
             for k in ('mIoU', 'IoU', 'MovingMicro')} for name in INPUTS}
    return dict(status='complete', windows=len(rows), scientific_updates=0, diagnostic_only=True,
        reports=reports, edits=edits, half_minus_original=delta, seconds=sum(r['seconds'] for r in rows),
        strengths=[1., .5], single_forward_per_modality=True, automatic_promotion=False,
        interpretation='Fixed DEV64 mechanism check, not independent test/FPS or proof of a learned gate. '
                       'Only final prior coefficient changes; refiner inputs/residual remain original.')


def evaluate(predictor, source, jobs, keys, contract, *, artifact=None, saved=None, save=None, stop_event=None):
    state = dict(contract=contract, rows=[]) if saved is None else copy.deepcopy(saved)
    volume = int(np.prod(predictor.provider.pcfg.grid.shape_hwd))
    checked_rows(state, keys, contract, volume)
    try:
        for i, job in enumerate(jobs):
            if i < len(state['rows']):
                if artifact:
                    _, raw, _ = source.history(job, 'stc')
                    for token, sem in zip(job.history_tokens, raw['history_occ']): artifact.check(job.scene_name, token, sem)
                continue
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('complete windows saved')
            tick = time.perf_counter(); row = row_report(predictor, source, job, artifact)
            row['seconds'] = time.perf_counter()-tick
            # Validate complete-window integrity BEFORE committing the row.
            checked_rows(dict(contract=contract, rows=[row]), [keys[i]], contract, volume)
            state['rows'].append(row)
            if save: save(state)
            if (i+1) % 8 == 0: print(f'PRIOR_STRENGTH {i+1}/{len(jobs)} seconds/window={row["seconds"]:.3f}', flush=True)
    finally:
        if save: save(state)
    if len(state['rows']) != len(keys) or not keys: raise RuntimeError('complete fixed population required')
    result = aggregate(state['rows'])
    if artifact: result['STC_artifact_audit'] = artifact.verify()
    return result
