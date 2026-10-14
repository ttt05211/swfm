import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('prior_report', Path(__file__).parents[1]/'tools/prior_strength_report.py')
reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)


def counts(pred, target):
    c = np.bincount(18*np.asarray(target)+np.asarray(pred), minlength=324).reshape(18, 18)
    return np.stack([c]*3)


def test_exact_damage_split_with_semantic_and_binary_errors():
    # free->occupied correct damage; occupied->free correct damage;
    # wrong occupancy->free correction; missed->wrong occupied (binary-only repair);
    # wrong occupied class->free (new binary miss but NOT correct-semantic damage).
    before = [17, 4, 4, 17, 4]; after = [11, 17, 17, 4, 17]; target = [17, 4, 17, 11, 15]
    edits = dict(all=dict(damaged=6), false_occupancy=dict(corrected=3), wrong_free=dict(changed=3))
    v = reader.damage_split(counts(before, target), counts(after, target), edits)
    assert v['damaged_correct_free'] == 3 and v['damaged_correct_occupied'] == 3
    assert v['newly_missed_occupancy'] == 6 and v['delta_FN'] == 3
    assert v['free_share_of_damage_pct'] == 50.


def test_damage_split_cannot_turn_net_fp_into_gross_damage():
    before = [4, 4, 17]; after = [17, 17, 4]; target = [17, 17, 17]
    edits = dict(all=dict(damaged=3), false_occupancy=dict(corrected=6), wrong_free=dict(changed=0))
    v = reader.damage_split(counts(before, target), counts(after, target), edits)
    assert v['delta_FP'] == -3 and v['damaged_correct_free'] == 3
    edits['false_occupancy']['corrected'] = 0
    with pytest.raises(ValueError, match='inconsistent'): reader.damage_split(counts(before, target), counts(after, target), edits)


def test_resolver_explicit_missing_does_not_silently_pick_other_run(tmp_path, monkeypatch):
    monkeypatch.setenv('PRIOR_DIAG_DIR', str(tmp_path/'explicit'))
    assert reader.resolve_result() == tmp_path/'explicit/evaluation.json'
    with pytest.raises(FileNotFoundError): reader.load(reader.resolve_result())


def test_full_read_only_report_reproduces_counts_and_rejects_corruption(tmp_path):
    target = np.array([17, 4, 17, 11, 15])
    original = np.array([17, 4, 4, 17, 4]); half = np.array([11, 17, 17, 4, 17])
    def edits(a, b):
        wrong, bad, changed = a != target, b != target, a != b
        masks = dict(all=np.ones(5, bool), wrong_free=(a == 17) & (target != 17),
            false_occupancy=(a != 17) & (target == 17), wrong_class=(a != 17) & (target != 17) & wrong)
        return {g: dict(count=int(m.sum()), changed=int((m & changed).sum()),
            corrected=int((m & wrong & ~bad).sum()), damaged=int((m & ~wrong & bad).sum()),
            wrong_to_other_wrong=int((m & wrong & bad & changed).sum())) for g, m in masks.items()}
    keys = [['scene'+str(i % 4), str(i)] for i in range(64)]
    contract = dict(protocol=reader.PROTOCOL, keys=keys)
    rows = []
    result = dict(status='complete', windows=64, scientific_updates=0, source_epoch=2, source_update=10292,
                  strengths=[1., .5], contract=contract, reports={}, edits={})
    c1, c05 = counts(original, target), counts(half, target)
    for key in keys:
        row = dict(key=key, counts={}, edits={})
        for name in reader.INPUTS:
            for label, c in zip(reader.LABELS, (c1, c1, c05)):
                row['counts'][name+'/'+label] = dict(confusion=c.tolist(), moving=np.zeros_like(c).tolist())
            row['edits'][name] = {label: [edits(a, b) for _ in range(3)] for label, a, b in (
                ('original_c1', original, original), ('diagnostic_c05', original, half), ('half_vs_original', original, half))}
        rows.append(row)
    for name in reader.INPUTS:
        for label, c in zip(reader.LABELS, (c1, c1, c05)):
            report = dict(average=dict(mIoU=reader.score(c), IoU=20.), per_horizon={})
            for h, m in zip(('1.0', '2.0', '3.0'), c):
                t = np.diag(m); u = m.sum(0)+m.sum(1)-t
                report['per_horizon'][h] = dict(mIoU=reader.score(c), IoU=20., semantic_per_class={str(i):
                    float(100*t[i]/u[i]) if u[i] else None for i in range(17)})
            result['reports'][name+'/'+label] = report
        result['edits'][name] = {label: dict(totals={g: {k: int(v)*192 for k, v in item.items()}
            for g, item in rows[0]['edits'][name][label][0].items()}) for label in rows[0]['edits'][name]}
    path = tmp_path/'evaluation.json'; state_path = tmp_path/'state.json'
    path.write_text(json.dumps(result)); state_path.write_text(json.dumps(dict(contract=contract, rows=rows)))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = (digest(path), digest(state_path))
    text = reader.report(path)
    assert 'damage_correct_free=' in text and 'scene-negative=' in text and 'worst5' in text
    assert before == (digest(path), digest(state_path))
    bad = copy.deepcopy(result); bad['edits']['stc_gt']['half_vs_original']['totals']['all']['damaged'] += 1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='totals'): reader.report(path)
