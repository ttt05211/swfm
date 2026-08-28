from types import SimpleNamespace

from tools.real_motion.p0_hybrid_scene_disjoint_validate import (
    select_scene_disjoint_windows,
)


class FakeSource:
    def __init__(self):
        self.nusc = SimpleNamespace(scene=[
            {"name": "scene-c"},
            {"name": "scene-a"},
            {"name": "scene-b"},
            {"name": "scene-x"},
        ])
        self.allowed_scenes = {"scene-a", "scene-b", "scene-c"}
        self._tokens = {
            "scene-a": [f"a{i}" for i in range(20)],
            "scene-b": [f"b{i}" for i in range(22)],
            "scene-c": [f"c{i}" for i in range(24)],
            "scene-x": [f"x{i}" for i in range(24)],
        }

    def scene_tokens(self, scene):
        return self._tokens[scene["name"]]


def test_scene_disjoint_selector_is_unique_and_reproducible():
    src = FakeSource()
    a = select_scene_disjoint_windows(src, history=6, future=6, max_windows=2, seed=7)
    b = select_scene_disjoint_windows(src, history=6, future=6, max_windows=2, seed=7)
    assert [w.scene_name for w in a] == [w.scene_name for w in b]
    assert len(a) == 2
    assert len({w.scene_name for w in a}) == 2
    assert all(w.scene_name in src.allowed_scenes for w in a)


def test_scene_disjoint_selector_uses_middle_eligible_window():
    src = FakeSource()
    windows = select_scene_disjoint_windows(
        src, history=6, future=6, max_windows=None, seed=0
    )
    by_scene = {w.scene_name: w for w in windows}

    # scene-a has 20 tokens. Eligible t0 indices are 5..13, midpoint is 9.
    w = by_scene["scene-a"]
    assert w.t0_token == "a9"
    assert w.history_tokens == tuple(f"a{i}" for i in range(4, 10))
    assert w.future_tokens == tuple(f"a{i}" for i in range(10, 16))


def test_scene_disjoint_selector_rejects_nonpositive_cap():
    src = FakeSource()
    try:
        select_scene_disjoint_windows(src, max_windows=0)
    except ValueError as exc:
        assert "max_windows" in str(exc)
    else:
        raise AssertionError("expected nonpositive max_windows to fail")
