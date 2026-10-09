"""Transport recovery must never change or partially publish the pinned pickle."""
import errno
import hashlib
import io
import os
import urllib.error

import pytest

from tools.real_motion import download_geniedrive_eval_info as downloader
from tools.real_motion import geniedrive_eval_alignment as genie


@pytest.fixture(autouse=True)
def fixed_test_artifact(monkeypatch):
    content = b'pinned test metadata'
    monkeypatch.delenv('GENIEDRIVE_DOWNLOAD_ENDPOINT', raising=False)
    monkeypatch.delenv('HF_ENDPOINT', raising=False)
    monkeypatch.setattr(genie, 'INFO_BYTES', len(content))
    monkeypatch.setattr(downloader, 'INFO_BYTES', len(content))
    monkeypatch.setattr(genie, 'INFO_SHA256', hashlib.sha256(content).hexdigest())
    return content


def test_network_unreachable_falls_back_to_same_pinned_path(tmp_path, monkeypatch, fixed_test_artifact):
    calls = []
    def open_request(request, timeout):
        calls.append((request.full_url, timeout))
        if len(calls) == 1:
            raise urllib.error.URLError(OSError(errno.ENETUNREACH, 'Network is unreachable'))
        assert not list(tmp_path.glob('*.part')) or all(p.stat().st_size == 0 for p in tmp_path.glob('*.part'))
        return io.BytesIO(fixed_test_artifact)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_request)
    path = tmp_path/'official.pkl'
    downloader.download(path, timeout=7)
    assert calls == [(downloader.INFO_URL, 7),
                     (downloader.INFO_URL.replace('https://huggingface.co', 'https://hf-mirror.com'), 7)]
    assert path.read_bytes() == fixed_test_artifact and not list(tmp_path.glob('*.part'))


def test_midstream_disconnect_removes_partial_before_fallback(tmp_path, monkeypatch, fixed_test_artifact):
    class BrokenStream(io.BytesIO):
        def read(self, size=-1):
            if self.tell(): raise ConnectionResetError('disconnected')
            return super().read(4)
    calls = []
    def open_request(request, timeout):
        calls.append(request.full_url)
        return BrokenStream(fixed_test_artifact) if len(calls) == 1 else io.BytesIO(fixed_test_artifact)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', open_request)
    path = tmp_path/'official.pkl'
    downloader.download(path)
    assert len(calls) == 2 and path.read_bytes() == fixed_test_artifact
    assert not list(tmp_path.glob('*.part'))


def test_custom_endpoint_precedence_preserves_revision(monkeypatch):
    monkeypatch.setenv('HF_ENDPOINT', 'https://hf-mirror.com')
    suffix = downloader.INFO_URL.split('https://huggingface.co', 1)[1]
    assert downloader.download_urls() == ('https://hf-mirror.com'+suffix,)
    monkeypatch.setenv('GENIEDRIVE_DOWNLOAD_ENDPOINT', 'https://example.org')
    assert downloader.download_urls() == ('https://example.org'+suffix,)
    assert downloader.download_urls('https://explicit.example/') == ('https://explicit.example'+suffix,)


@pytest.mark.parametrize('origin', ['http://example.org', 'https://u:p@example.org',
    'https://example.org/other', 'https://example.org?q=1', 'https://example.org#f', 'https://'])
def test_invalid_endpoint_rejected_without_network(origin, monkeypatch, tmp_path):
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network'))
    with pytest.raises(ValueError, match='HTTPS origin'):
        downloader.download(tmp_path/'official.pkl', endpoint=origin)
    assert not list(tmp_path.iterdir())


def test_all_network_failures_clean_and_report_offline_path(tmp_path, monkeypatch):
    def unreachable(*args, **kwargs): raise urllib.error.URLError('offline')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', unreachable)
    path = tmp_path/'official.pkl'
    with pytest.raises(downloader.DownloadNetworkError, match='评估尚未启动') as error:
        downloader.download(path)
    assert str(path.resolve()) in str(error.value)
    assert 'huggingface.co' in str(error.value) and 'hf-mirror.com' in str(error.value)
    assert not path.exists() and not list(tmp_path.glob('*.part'))


def test_integrity_failure_is_not_retried_or_published(tmp_path, monkeypatch, fixed_test_artifact):
    calls = []
    def corrupt(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'X'*len(fixed_test_artifact))
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', corrupt)
    path = tmp_path/'official.pkl'
    with pytest.raises(RuntimeError, match='SHA256 mismatch'): downloader.download(path)
    assert len(calls) == 1 and not path.exists() and not list(tmp_path.glob('*.part'))


def test_existing_verified_artifact_reused_offline(tmp_path, monkeypatch, fixed_test_artifact):
    path = tmp_path/'official.pkl'; path.write_bytes(fixed_test_artifact)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network'))
    assert downloader.download(path) == path.resolve()
    path.write_bytes(b'X'*len(fixed_test_artifact))
    with pytest.raises(RuntimeError, match='SHA256 mismatch'): downloader.download(path)
    assert path.read_bytes() == b'X'*len(fixed_test_artifact)


def test_other_process_temporary_file_is_not_removed(tmp_path, monkeypatch):
    path = tmp_path/'official.pkl'
    temporary = tmp_path/f'official.pkl.{os.getpid()}.part'
    temporary.write_bytes(b'not ours')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network'))
    with pytest.raises(FileExistsError): downloader.download(path)
    assert temporary.read_bytes() == b'not ours'


def test_cli_reports_network_failure_without_traceback(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(downloader.sys, 'argv', ['download', '--out', str(tmp_path/'official.pkl')])
    def offline(*args, **kwargs): raise downloader.DownloadNetworkError('评估尚未启动：offline')
    monkeypatch.setattr(downloader, 'download', offline)
    with pytest.raises(SystemExit) as exit_error: downloader.main()
    assert exit_error.value.code == 2
    stderr = capsys.readouterr().err
    assert '评估尚未启动' in stderr and 'Traceback' not in stderr
