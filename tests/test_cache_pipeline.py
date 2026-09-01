import time

import pytest

from real_motion.cache_pipeline import bounded_ordered_parallel_map


def test_bounded_parallel_map_preserves_input_order():
    def work(x):
        # Finish later inputs earlier to ensure completion order differs.
        time.sleep((5 - x) * 0.001)
        return x * 10

    out = list(bounded_ordered_parallel_map(
        work,
        range(6),
        max_workers=3,
        max_in_flight=4,
    ))
    assert out == [0, 10, 20, 30, 40, 50]


def test_bounded_parallel_map_serial_fallback_and_validation():
    assert list(bounded_ordered_parallel_map(
        lambda x: x + 1,
        [1, 2, 3],
        max_workers=1,
        max_in_flight=1,
    )) == [2, 3, 4]
    with pytest.raises(ValueError):
        list(bounded_ordered_parallel_map(lambda x: x, [1], max_workers=0, max_in_flight=1))
    with pytest.raises(ValueError):
        list(bounded_ordered_parallel_map(lambda x: x, [1], max_workers=1, max_in_flight=0))
