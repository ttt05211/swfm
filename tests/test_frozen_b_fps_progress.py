from tools.real_motion.benchmark_p0_f9_dense_forecast_fps_frozen_b import _aggregate


def _row(i,repeat,seconds=.1):
    return {
        "key":f"scene/token{i}",
        "repeat":repeat,
        "seconds":seconds,
        "peak_allocated_mib":100.0,
        "incremental_peak_allocated_mib":10.0,
        "peak_reserved_mib":120.0,
        "host_stages_seconds":{"x":.01},
    }


def test_partial_aggregate_is_allowed_for_progress_reporting():
    rows=[_row(0,1),_row(0,2),_row(0,3)]
    out=_aggregate(rows)
    assert out["samples"]==3
    assert out["windows"]==1
    assert abs(out["dense_forecast_fps"]-60.0)<1e-12
