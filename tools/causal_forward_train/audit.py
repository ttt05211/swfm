"""CPU GT-box audit AFTER old/new links are fixed; no model or training."""
from collections import Counter
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import signal
import threading
import numpy as np
from tools.joint_occ_experiments import run as base
from tools.source_scene_stage1.association import attribute,aggregate,boxes_for_sample,PURITY,SECONDARY
from tools.source_scene_stage2.track import build as old_build
from tools.causal_forward_track.track import build as new_build,PROTOCOL as TRACK_PROTOCOL
from tools.real_motion import benchmark_p0_f9_v18_runtime as runtime
from real_motion.nuscenes_adapter import sample_ego_to_world
from real_motion.strong_w2det import StrongW2DetConfig
from .control import reference,sha

PROTOCOL = 'same64_original_forward_GT_box_identity_audit_v1'


def inspect(identity,frames,poses,times,boxes,grid):
    # identity is a completed prediction; this function never changes its links.
    pa = attribute(frames[-2],boxes[-2],poses[-2],grid)
    ca = attribute(frames[-1],boxes[-1],poses[-1],grid)
    before,now = ({b['instance_token']:b for b in bs} for bs in (boxes[-2],boxes[-1]))
    eligible = {a['token'] for a in pa if a['token']} & {a['token'] for a in ca if a['token']}
    recovered = set();edges = []
    for i,j in enumerate(identity.links[:,2]):
        c = ca[i];p = pa[int(j)] if j >= 0 else None
        resolved = bool(p and c['token'] is not None and p['token'] is not None)
        correct = bool(resolved and c['token'] == p['token'])
        local = (np.asarray(frames[-1][i]['centroid_world'])-poses[-1,:3,3])@poses[-1,:3,:3]
        e = dict(current=i,previous=int(j),class_id=int(frames[-1][i]['class_id']),range_m=float(np.linalg.norm(local[:2])),
            current_attribution=c,previous_attribution=p,resolved=resolved,correct=correct,
            status='unmatched' if j < 0 else 'correct' if correct else 'wrong_identity' if resolved else 'unresolved')
        token = c['token']
        if token in before and token in now:
            velocity = (np.asarray(now[token]['center'])-np.asarray(before[token]['center']))/(times[-1]-times[-2])
            e['GT_speed_mps'] = float(np.linalg.norm(velocity[:2]))
            if correct:
                e['KTA_velocity_error_mps'] = float(np.linalg.norm(identity.pair_velocity[i][:2]-velocity[:2]))
        if correct:recovered.add(token)
        edges.append(e)
    return dict(edges=edges,eligible_instances=sorted(eligible),recovered_instances=sorted(recovered))


def subset(row,indices):
    edges = [row['edges'][i] for i in indices]
    tokens = {e['current_attribution']['token'] for e in edges}
    return dict(edges=edges,eligible_instances=sorted(set(row['eligible_instances'])&tokens),
        recovered_instances=sorted({e['current_attribution']['token'] for e in edges if e['correct']}))


def reports(rows):
    output = {}
    for kind in ('occ','stc'):
        values = [r[kind] for r in rows]
        transitions = Counter();both = Counter()
        for r in values:
            for i in r['changed']:
                a,b = r['baseline']['edges'][i],r['forward']['edges'][i]
                transitions[a['status']+' -> '+b['status']] += 1
                if a['resolved'] and b['resolved']:
                    both['sources'] += 1
                    both['old_correct'] += int(a['correct']);both['new_correct'] += int(b['correct'])
                    both['wrong_to_correct'] += int(not a['correct'] and b['correct'])
                    both['correct_to_wrong'] += int(a['correct'] and not b['correct'])
        output[kind] = dict(all={arm:aggregate([r[arm] for r in values]) for arm in ('baseline','forward')},
            changed_only={arm:aggregate([subset(r[arm],r['changed']) for r in values]) for arm in ('baseline','forward')},
            changed_status_transitions=dict(transitions),changed_both_resolved=dict(both))
    return output


def summarize(value):
    lines = ['===== SAME64 / ORIGINAL vs FORWARD / GT-BOX PAIR AUDIT =====',
        f'status={value["status"]}; windows={len(value["rows"])}/64; CPU only; no model/optimizer.',
        'GT boxes label already-completed links; never enter either matcher. No registration.',
        'Conditional precision/coverage/identifiable recall/GT-speed strata; changed-only is also conditional.']
    for kind,m in value['reports'].items():
        for population in ('all','changed_only'):
            for arm,row in m[population].items():
                lines.append(f'{kind}/{population}/{arm} '+json.dumps(row,ensure_ascii=False))
        lines.append(kind+'/changed_status_transitions '+json.dumps(m['changed_status_transitions']))
        lines.append(kind+'/changed_both_resolved '+json.dumps(m['changed_both_resolved']))
    return '\n'.join(lines)+'\n'


def main(argv=None,event=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('control-dir','dataroot','dev-cache','stc-root','dev-info','out-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--config',default='configs/real_motion_occfm.yaml');p.add_argument('--resume',action='store_true')
    a = p.parse_args(argv);event = event or threading.Event();cp,control = reference(a.control_dir)
    pcfg = base.make_prepare_config(base.load_runtime_config(a.config,control['contract']['overrides']))
    strong = StrongW2DetConfig(free_label=17)
    if pcfg.frame_dt_s != .5:raise RuntimeError('original nominal .5s KTA required')
    for name in ('dev_cache','dev_info','config'):
        if sha(getattr(a,name)) != control['contract']['data'][name]:raise RuntimeError('control data differs: '+name)
    _,records = base.load_cache(a.dev_cache)
    chosen = base.align_records(records,control['contract']['eval_keys'])
    from nuscenes.nuscenes import NuScenes
    data = Path(a.dataroot).resolve();stc = Path(a.stc_root).resolve()
    nusc = NuScenes(version='v1.0-trainval',dataroot=str(data),verbose=False)
    inventory = set()
    for r in chosen:
        for token in r['history_tokens'][-4:]:
            for root in (data/'gts',stc):
                path = (root/r['scene_name']/str(token)/'labels.npz').resolve();stat = path.stat()
                inventory.add((str(path),stat.st_size,stat.st_mtime_ns))
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__),Path(__file__).with_name('control.py'),root/'tools/causal_forward_track/track.py',
        root/'tools/source_scene_stage2/track.py',root/'tools/source_scene_stage1/association.py',Path(runtime.__file__),
        root/'real_motion/shared_source_track.py']
    contract = json.loads(json.dumps(dict(protocol=PROTOCOL,track_protocol=TRACK_PROTOCOL,keys=control['contract']['eval_keys'],
        control_results_sha256=sha(cp),strong=asdict(strong),grid=asdict(pcfg.grid),files=sorted(inventory),
        attribution=dict(purity=PURITY,secondary=SECONDARY),
        metadata={name:sha(data/'v1.0-trainval'/name) for name in ('sample_annotation.json','sample.json','instance.json',
            'category.json','sample_data.json','ego_pose.json','calibrated_sensor.json')},
        implementation={f.relative_to(root).as_posix():sha(f) for f in files})))
    out = Path(a.out_dir).resolve();rows = []
    from tools.real_motion.eval_p0_f9_joint_checkpoints import evaluation_lock
    lock = out.parent/'.forward_audit_locks'/base.stable_json_fingerprint(str(out));lock.mkdir(parents=True,exist_ok=True)
    with evaluation_lock(lock):
        if a.resume:
            saved = json.loads((out/'results.json').read_text(encoding='utf-8'))
            if saved['contract'] != contract:raise RuntimeError('audit provenance changed')
            rows = saved['rows']
            if [r['key'] for r in rows] != contract['keys'][:len(rows)]:raise RuntimeError('invalid complete-window prefix')
        else:out.mkdir(parents=True,exist_ok=False)
        def publish(status):
            value = dict(status=status,contract=contract,rows=rows,reports=reports(rows))
            base.write_json(out/'results.json',value);(out/'summary.txt').write_text(summarize(value),encoding='utf-8')
            return value
        try:
            for r in chosen[len(rows):]:
                if event.is_set():break
                tokens = list(map(str,r['history_tokens'][-4:]))
                poses = np.asarray([sample_ego_to_world(nusc,t) for t in tokens])
                times = np.asarray([nusc.get('sample',t)['timestamp']/1e6 for t in tokens])
                completed = {}
                for kind,folder in (('occ',data/'gts'),('stc',stc)):
                    frames = []
                    for token,pose in zip(tokens,poses):
                        with np.load(folder/r['scene_name']/token/'labels.npz',allow_pickle=False) as z:
                            sem = z['semantics']
                            if (sem.shape != tuple(pcfg.grid.shape_hwd) or sem.dtype.kind not in 'iu'
                                    or np.any((sem < 0)|(sem > 17))):raise ValueError('invalid history semantics')
                            frames.append(runtime.extract_instances_cropped_exact(sem,pose,grid=pcfg.grid,cfg=strong))
                    completed[kind] = frames,old_build(frames,poses,times,strong),new_build(frames,poses,times,strong)
                # Only now load GT instance annotations. No feedback to association.
                boxes = [boxes_for_sample(nusc,t) for t in tokens]
                item = dict(key=[r['scene_name'],r['t0_token']])
                for kind,(frames,old,new) in completed.items():
                    item[kind] = dict(baseline=inspect(old,frames,poses,times,boxes,pcfg.grid),
                        forward=inspect(new,frames,poses,times,boxes,pcfg.grid),
                        changed=np.flatnonzero(old.links[:,2] != new.links[:,2]).tolist())
                rows.append(item);publish('running')
                if len(rows)%8 == 0:print(f'FORWARD_PAIR_AUDIT {len(rows)}/64',flush=True)
        finally:result = publish('complete' if len(rows)==64 else 'stopped')
    print(summarize(result),flush=True)
    return 0 if result['status']=='complete' else 130


if __name__ == '__main__':
    event = threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:event.set())
    raise SystemExit(main(event=event))
