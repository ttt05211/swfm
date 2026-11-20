"""Evaluation identity metadata contains no GT TRAIN source-budget features."""
import argparse
import json
from pathlib import Path
from .run import PROTOCOL, prepare_row

FIELDS = ('scene_name', 't0_token', 'history_tokens', 'future_tokens')


def prepare_evaluation_row(inputs, predictor, record, *, labels=False, kind='occ'):
    if labels or kind not in ('occ', 'stc'):
        raise ValueError('evaluation accepts historical OCC/STC only; labels follow six predictions')
    # Cached `features` counts GT components for TRAIN source128 grouping.
    # STC derives its OWN components from its four inputs. Nothing in the
    # inference path consumes the omitted features; only the old guard did.
    identity = {name: record[name] for name in FIELDS}
    return prepare_row(inputs, predictor, identity, labels=False, kind=kind)


def select_interrupted_run(root):
    """Choose a unique completed model with unfinished evaluation, never latest/best."""
    choices = []
    for folder in Path(root).glob('coverage_projection_*'):
        try:
            if not folder.is_dir() or not (folder/'epoch_0001.pt').is_file(): continue
            c = json.loads((folder/'contract.json').read_text(encoding='utf-8'))
            progress = json.loads((folder/'progress.json').read_text(encoding='utf-8'))
            state = json.loads((folder/'evaluation_state.json').read_text(encoding='utf-8'))
            routes = {'control/occ', 'control/stc', 'candidate/occ', 'candidate/stc'}
            completed = state['progress']
            if (c.get('protocol') != PROTOCOL or progress.get('updates') != 1031
                    or progress.get('status') != 'complete' or len(c.get('eval_keys', [])) != 64
                    or state.get('receipt', {}).get('keys') != c['eval_keys']
                    or not set(completed) <= routes
                    or any(type(n) is not int or not 0 <= n <= 64 for n in completed.values())
                    or all(completed.get(r, 0) == 64 for r in routes)):
                continue
            choices.append(folder.resolve())
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if len(choices) != 1:
        raise ValueError('Set COVERAGE_DIR to the original completed training run; '
            'no date/score selection. Matching interrupted runs='+str(choices))
    return choices[0]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--find-run', required=True, type=Path)
    print(select_interrupted_run(p.parse_args().find_run))
