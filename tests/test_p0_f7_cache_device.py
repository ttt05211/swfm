import torch

from tools.real_motion.build_p0_f7_cache_fast import _resolve_device


def test_bare_cuda_uses_current_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert _resolve_device("cuda") == torch.device("cuda:3")


def test_explicit_cuda_index_is_preserved(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert _resolve_device("cuda:2") == torch.device("cuda:2")


def test_cuda_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device("cuda") == torch.device("cpu")
