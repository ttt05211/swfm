import json
from pathlib import Path
import pytest
from tools import source_motion_audit_brief as report


def fixture():
    routes = {m+'/'+r: dict(mIoU=20., IoU=30., MovingMicro=10.) for m in report.MODES for r in report.READOUTS}
    routes['learned/transport']['mIoU'] = 20.125
    return dict(status='complete', windows=64, metrics={'occ_gt': routes, 'stc_gt': routes}, source_motion={})


def test_compact_preserves_all_fixed_routes_and_missing_stats():
    text = report.brief(fixture(), Path('results.json'))
    assert '64/64' in text and '+0.1250' in text and 'NA' in text
    assert len(text.splitlines()) <= 30
    for mode in report.MODES: assert text.count(mode) == 2
    assert '单窗最大差异' in text and 'OCC+GT' in text and 'STC+GT' in text


def test_partial_and_no_windows_not_presented_as_complete():
    data = fixture(); data.update(status='running', windows=8)
    assert '不是最终64窗结果' in report.brief(data, Path('x'))
    data['metrics'] = {}
    assert '暂无已保存指标' in report.brief(data, Path('x'))
    assert report.number(None) == report.number(float('nan')) == report.number(float('inf')) == 'NA'
    assert report.delta(None, 2.) == 'NA'


def test_explicit_and_env_paths_are_read_only(tmp_path, monkeypatch, capsys):
    path = tmp_path/'results.json'; path.write_text(json.dumps(fixture()), encoding='utf-8')
    before = path.read_bytes()
    monkeypatch.setenv('MOTION_AUDIT_DIR', str(tmp_path))
    assert report.find_result() == path
    assert report.find_result(str(path)) == path
    report.main([str(tmp_path)])
    assert '20.1250' in capsys.readouterr().out and path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_wrong_file_errors_without_traceback(tmp_path, capsys):
    path = tmp_path/'state.json'; path.write_text('{}')
    with pytest.raises(SystemExit) as exc: report.main([str(path)])
    assert exc.value.code == 2 and 'metrics' in capsys.readouterr().err
