"""Join same64 A, fresh A/B and C without selecting a model or hiding drift."""
import argparse
import json
from pathlib import Path
from .control import reference,sha,NAMES


def combine(control_dir,root):
    cp,a = reference(control_dir);root = Path(root)
    b = json.loads((root/'frozen_B/results.json').read_text(encoding='utf-8'))
    c = json.loads((root/'C_train/results.json').read_text(encoding='utf-8'))
    audit = json.loads((root/'pair_audit/results.json').read_text(encoding='utf-8'))
    keys = a['contract']['eval_keys']
    from .run import PROTOCOL
    from .model import SOURCE_POLICY
    from .audit import PROTOCOL as AUDIT_PROTOCOL
    from tools.causal_forward_track.run import PROTOCOL as FROZEN_PROTOCOL
    if (b['status'] != 'complete' or b['keys'] != keys or b['contract']['keys'] != keys
            or b['contract']['protocol'] != FROZEN_PROTOCOL
            or b['contract']['stc_age'] or b['contract']['horizon_chunk'] != 6
            or b['contract']['source_contract'] != a['contract']
            or b['contract']['source']['checkpoint_sha256'] != sha(cp.parent/'epoch_0001.pt')):
        raise RuntimeError('B must use the exact A checkpoint/same64/H6/no age')
    fields = ('train_keys','eval_keys','epochs','total_updates','seed','objective','learning_rates',
        'schedule','known','precision','world_size','data','overrides','STC_inventory','execution','parameters')
    if (c['status'] != 'complete' or c['epoch'] != 1 or c['contract']['protocol'] != PROTOCOL
            or c['contract']['source_policy'] != SOURCE_POLICY or c['evaluations']['1']['keys'] != keys
            or c['contract']['model']['experiment']['control_results_sha256'] != sha(cp)):
        raise RuntimeError('C must be completed matched forward TRAIN4086x1')
    for field in fields:
        if c['contract'][field] != a['contract'][field]:raise RuntimeError('C differs from A: '+field)
    if (audit['status'] != 'complete' or len(audit['rows']) != 64 or audit['contract']['keys'] != keys
            or audit['contract']['protocol'] != AUDIT_PROTOCOL or [r['key'] for r in audit['rows']] != keys
            or audit['contract']['control_results_sha256'] != sha(cp)):
        raise RuntimeError('same64 complete pair audit required')
    rows = {};deltas = {}
    for name in NAMES:
        kind,route = name.split('_gt_',1);old = a['evaluations']['1']['variants'][name]
        values = dict(A_recorded=old,
            A_fresh=b['reports'][kind+'/baseline/'+route]['average'],
            B_old_weights_forward=b['reports'][kind+'/forward/'+route]['average'],
            C_forward_trained=c['evaluations']['1']['variants'][name])
        rows[name] = values
        deltas[name] = {label:{k:values[x][k]-values[y][k] for k in ('mIoU','IoU','MovingMicro')}
            for label,x,y in (('fresh_A_minus_recorded','A_fresh','A_recorded'),
                ('B_minus_fresh_A_input_policy','B_old_weights_forward','A_fresh'),
                ('C_minus_B_training_policy','C_forward_trained','B_old_weights_forward'),
                ('C_minus_recorded_A_total','C_forward_trained','A_recorded'))}
    result = dict(status='complete',keys=keys,rows=rows,deltas=deltas,pair_audit=audit['reports'],
        provenance=dict(control_results_sha256=sha(cp),B_results_sha256=sha(root/'frozen_B/results.json'),
            C_results_sha256=sha(root/'C_train/results.json'),audit_results_sha256=sha(root/'pair_audit/results.json')))
    lines = ['===== MATCHED AdaLN-Zero / HISTORY ASSOCIATION / A-B-C =====',
        'status=complete; SAME TRAIN4086x1/order/seed/budget/initial weights/architecture/same64.',
        'A recorded=old train/old eval. A fresh+B=ONE frozen old checkpoint, old vs new eval.',
        'C=new RANDOM training with forward association in BOTH TRAIN and EVAL; no trained weight import.',
        'GT future ego/dense-valid/H6/full-grid/CE+Lovasz only; no age/bias/calibration/head/noise.',
        'route model mIoU IoU MovingMicro']
    for name,values in rows.items():
        for label,m in values.items():
            lines.append(f'{name:28s} {label:24s} {m["mIoU"]:.6f} {m["IoU"]:.6f} {m["MovingMicro"]:.6f}')
        for label,delta in deltas[name].items():lines.append(name+' '+label+'='+json.dumps(delta))
    lines += ['Pair audit: pair_audit/summary.txt (all AND changed-only precision/coverage/recall/GT-speed strata).',
        'Repeated development64; conditional GT-box attribution is not perfect instance truth.',
        'All fixed final results retained; no best selection/threshold tuning/full training/promotion. NOT FPS.']
    (root/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-dir',required=True);p.add_argument('--out-dir',required=True)
    a = p.parse_args();print(combine(a.control_dir,a.out_dir))
