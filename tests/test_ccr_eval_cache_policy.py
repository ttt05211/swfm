from types import SimpleNamespace

from tools.real_motion.ccr_screen_common import _evaluation_geometry, _ccr_minimal_fixed_geometry


def test_old_local_eval_bypasses_slim_val_cache_and_restores_state():
    dev_source=object()
    original_builder=object()
    provider=SimpleNamespace(
        ccr_fast_train=True,
        fixed_geometry_builder=original_builder,
        ccr_val_history_cache=object(),
        ccr_val_history_cache_source=dev_source,
    )
    with _evaluation_geometry(provider,include_old=True):
        assert provider.fixed_geometry_builder is None
        assert provider.ccr_val_history_cache_source is not dev_source
    assert provider.fixed_geometry_builder is original_builder
    assert provider.ccr_val_history_cache_source is dev_source


def test_current_only_eval_keeps_val_cache_and_uses_minimal_builder():
    dev_source=object()
    original_builder=object()
    provider=SimpleNamespace(
        ccr_fast_train=True,
        fixed_geometry_builder=original_builder,
        ccr_val_history_cache=object(),
        ccr_val_history_cache_source=dev_source,
    )
    with _evaluation_geometry(provider,include_old=False):
        assert provider.fixed_geometry_builder is _ccr_minimal_fixed_geometry
        assert provider.ccr_val_history_cache_source is dev_source
    assert provider.fixed_geometry_builder is original_builder
    assert provider.ccr_val_history_cache_source is dev_source
