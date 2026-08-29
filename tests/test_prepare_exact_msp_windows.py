import pytest

from tools.real_motion.prepare_exact_msp_windows import _window_from_record


def _record():
    hist = tuple(f"h{i}" for i in range(5)) + ("t0",)
    return {
        "sample_id": "scene-1:t0",
        "scene_name": "scene-1",
        "history_tokens": hist,
        "t0_token": "t0",
        "future_tokens": tuple(f"f{i}" for i in range(6)),
    }


def test_exact_window_reconstructed_verbatim():
    r = _record()
    w = _window_from_record(r, history_frames=6, future_frames=6)
    assert w.scene_name == r["scene_name"]
    assert w.history_tokens == r["history_tokens"]
    assert w.t0_token == r["t0_token"]
    assert w.future_tokens == r["future_tokens"]


def test_exact_window_rejects_mismatched_sample_id():
    r = _record()
    r["sample_id"] = "scene-1:other"
    with pytest.raises(RuntimeError, match="sample_id mismatch"):
        _window_from_record(r, history_frames=6, future_frames=6)


def test_exact_window_rejects_history_not_ending_at_t0():
    r = _record()
    r["history_tokens"] = tuple(f"h{i}" for i in range(6))
    with pytest.raises(RuntimeError, match="history does not terminate"):
        _window_from_record(r, history_frames=6, future_frames=6)
