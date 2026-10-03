"""Frozen population extraction must work after JSON serialization, without fallback."""
import copy
import json

import pytest

from tools.real_motion.eval_p0_f9_v21_stage0_upper_bounds import align_records


def test_json_dev512_is_extracted_from_full4369_in_manifest_order():
    records = [dict(scene_name=f'scene-{i//64}', t0_token=f'token-{i}') for i in range(4369)]
    indices = list(range(71, 3655, 7))[::-1]
    assert len(indices) == 512
    manifest = json.loads(json.dumps(dict(parent_keys=[
        (records[i]['scene_name'], records[i]['t0_token']) for i in indices])))
    original = copy.deepcopy(manifest)
    selected = align_records(list(reversed(records)), manifest['parent_keys'])
    assert len(selected) == 512
    assert all(record is records[i] for record, i in zip(selected, indices))
    assert manifest == original  # Do not mutate the fingerprinted manifest.


def test_alignment_accepts_mixed_list_tuple_pairs_and_single_pass_iterator():
    records = [dict(scene_name='s', t0_token='a'), dict(scene_name='s', t0_token='b')]
    assert align_records(records, iter([['s', 'b'], ('s', 'a')])) == records[::-1]


@pytest.mark.parametrize('bad_key', [None, 'sa', {'scene': 's', 'token': 'a'},
                                    ['s'], ['s', 'a', 'extra'], ['s', ['a']], ['', 'a'], ['s', 1]])
def test_malformed_population_identity_fails_closed(bad_key):
    with pytest.raises(RuntimeError, match='invalid population identity'):
        align_records([], [bad_key])


def test_duplicate_population_identity_fails_closed_after_list_normalization():
    records = [dict(scene_name='s', t0_token='a')]
    with pytest.raises(RuntimeError, match='duplicate population identities'):
        align_records(records, [['s', 'a'], ('s', 'a')])


def test_missing_identity_never_falls_back_to_cache_prefix():
    records = [dict(scene_name='s', t0_token='a')]
    with pytest.raises(RuntimeError, match='V18 cache missing keys'):
        align_records(records, [['s', 'missing']])


def test_duplicate_cache_identity_outside_selection_still_fails_closed():
    records = [dict(scene_name='s', t0_token=token) for token in ('a', 'b', 'b')]
    with pytest.raises(RuntimeError, match='V18 cache has duplicate identities'):
        align_records(records, [['s', 'a']])
