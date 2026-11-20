"""Pin the completed joint run: metadata/scores only, never a weight initializer."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.coverage_projection.run import ROOT, PROTOCOL as JOINT_PROTOCOL, sha
from tools.coverage_projection.control import reference as original_reference
from tools.coverage_projection.projection import PROJECTION
from tools.final_semantic_evaluation import metrics

PROTOCOL = 'forward_coverage_prior_ONLY_RANDOM_TRAIN4086x1_dual_v1'
OBJECTIVE = dict(CE=1., Lovasz=1., projection=0.)


def canonical(value): return json.loads(json.dumps(value))


def verify_files(implementation):
    for name, digest in implementation.items():
        path = (ROOT/name).resolve()
        if ROOT not in path.parents or sha(path) != digest:
            raise ValueError('frozen implementation differs: '+name)


def validate_joint(saved):
    c = canonical(saved['contract'])
    expected = dict(protocol=JOINT_PROTOCOL, total_updates=1031, epochs=1, world_size=2,
        parameters=2816554, seed=20261011, learning_rates=[5e-4,3e-4],
        schedule='whole_1_epoch_cosine_floor0.01', known='TRAIN_OCC_STC_dense_valid',
        source_policy='causal_forward_CV_history', registration='none',
        objective=dict(CE=1.,Lovasz=1.,projection=PROJECTION))
    for key, value in expected.items():
        if c.get(key) != value: raise ValueError(f'joint contract.{key} differs; actual={c.get(key)!r}')
    if saved.get('protocol') != JOINT_PROTOCOL or saved.get('updates') != 1031:
        raise ValueError('completed joint epoch1/update1031 required')
    for name, count in (('train_keys',4086),('eval_keys',64)):
        if len(c.get(name,[])) != count or len(set(map(tuple,c[name]))) != count:
            raise ValueError('joint population differs: '+name)
    flat = [tuple(k) for group in c.get('group_keys',[]) for k in group]
    if (len(c.get('group_keys',[])) != 1031 or len(flat) != 4086 or len(set(flat)) != 4086
            or set(flat) != set(map(tuple,c['train_keys']))
            or any(not 1 <= len(g) <= 4 for g in c['group_keys'])):
        raise ValueError('joint global window groups are incomplete/duplicated')
    if (c['model'].get('variant') != 'forward_FPN_AdaLN_UNet_coverage_projection_v1'
            or c['model'].get('source_policy') != c['source_policy']):
        raise ValueError('latest coverage-aware model architecture required')
    return c


def validate_evaluation(result, state, c, checkpoint_sha):
    receipt = result.get('receipt',{})
    if (result.get('status') != 'complete' or receipt.get('candidate') != checkpoint_sha
            or receipt.get('control') != c['control_sha256'] or receipt.get('keys') != c['eval_keys']
            or receipt.get('known') != 'dense_valid' or receipt.get('full_grid') is not True
            or receipt.get('horizons') != [1.,2.,3.] or receipt.get('indices') != [1,3,5]
            or any(receipt.get(k) is not False for k in ('FREE','fixed_age','calibration'))
            or state.get('receipt') != receipt):
        raise ValueError('complete compatible joint SAME64 full-grid evaluation required')
    routes = {f'{model}/{kind}' for model in ('control','candidate') for kind in ('occ','stc')}
    if set(state.get('progress',{})) != routes or any(v != 64 for v in state['progress'].values()):
        raise ValueError('joint metrics need complete original+joint OCC/STC64')
    if set(result.get('reports',{})) != routes or set(state.get('counts',{})) != routes:
        raise ValueError('joint report routes differ')
    for name in routes:
        if set(state['counts'][name]) != {'transport','semantic_refine'}: raise ValueError('joint readouts differ')
        for route, raw in state['counts'][name].items():
            values = {k:np.asarray(v) for k,v in raw.items()}
            if (set(values) != {'confusion','moving'} or any(a.shape != (3,18,18) or a.dtype.kind not in 'iu'
                    or np.any(a < 0) for a in values.values())
                    or not np.all(values['confusion'].sum((1,2)) == 64*200*200*16)):
                raise ValueError('joint full-grid integer counts invalid')
            computed = metrics.report(values,(1.,2.,3.),moving=False)
            if computed != result['reports'][name][route]: raise ValueError('joint scores differ from integer counts')
    return result


def reference(folder):
    folder = Path(folder).resolve(); checkpoint = folder/'epoch_0001.pt'
    digest = sha(checkpoint)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    c = validate_joint(saved); del saved  # Trained tensors are NEVER copied into the candidate.
    if sha(checkpoint) != digest: raise ValueError('joint checkpoint changed while reading')
    disk_contract = json.loads((folder/'contract.json').read_text(encoding='utf-8'))
    if canonical(disk_contract) != c: raise ValueError('joint checkpoint/disk contract differ')
    verify_files(c['implementations'])
    original = original_reference(c['control_dir'])
    if (sha(Path(c['control_dir'])/'epoch_0001.pt') != c['control_sha256']
            or sha(Path(c['control_dir'])/'results.json') != c['control_results_sha256']):
        raise ValueError('original matched control changed')
    for key in ('train_keys','eval_keys','seed','learning_rates','schedule','known','source_policy'):
        if original['contract'][key] != c[key]: raise ValueError('joint/original setting differs: '+key)
    evaluation = json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    state = json.loads((folder/'evaluation_state.json').read_text(encoding='utf-8'))
    validate_evaluation(evaluation,state,c,digest)
    return dict(folder=folder,contract=c,original_contract=original['contract'],evaluation=evaluation,
        checkpoint_sha256=digest,contract_sha256=sha(folder/'contract.json'),
        evaluation_sha256=sha(folder/'evaluation.json'))


def validate_completed(saved):
    c = saved['contract']
    if (saved.get('protocol') != PROTOCOL or c.get('protocol') != PROTOCOL
            or saved.get('updates') != 1031 or saved.get('epoch') != 1 or saved.get('batch') != 0
            or c.get('objective') != OBJECTIVE or c.get('total_updates') != 1031
            or c.get('world_size') != 2 or c.get('parameters') != 2816554):
        raise ValueError('completed coverage/Prior-only epoch1 required; no joint/partial fallback')
    verify_files(c['implementations'])
    return c


def select_run(root):
    choices=[]
    for folder in Path(root).glob('coverage_prior_only_*'):
        try:
            c=json.loads((folder/'contract.json').read_text());p=json.loads((folder/'progress.json').read_text())
            if c.get('protocol') == PROTOCOL and p.get('updates') == 1031 and (folder/'epoch_0001.pt').is_file():
                if not (folder/'evaluation.json').is_file(): choices.append(folder.resolve())
        except (OSError,ValueError,KeyError): continue
    if len(choices) != 1: raise ValueError('Set PRIOR_ONLY_DIR; no date/score selection. Matching runs='+str(choices))
    return choices[0]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--joint-dir',type=Path);p.add_argument('--find-run',type=Path)
    a=p.parse_args()
    if a.find_run: print(select_run(a.find_run))
    elif a.joint_dir:
        value=reference(a.joint_dir)
        print('PRIOR_ONLY_REFERENCE '+json.dumps(dict(joint_dir=str(value['folder']),updates=1031,
            joint_sha256=value['checkpoint_sha256'],training_weight_import=False,metrics='same64 complete original+joint reused')))
    else: p.error('pass --joint-dir or --find-run')
