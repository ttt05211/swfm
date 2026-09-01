"""Deterministic helpers for high-throughput cache construction.

Cache preparation contains large CPU-side occupancy arrays.  A bounded thread
pool lets file I/O and NumPy/SciPy native kernels overlap with GPU VAE encoding
without the huge IPC copies a process pool would require.  Results are yielded
in input order so cache sample/shard ordering remains deterministic.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def bounded_ordered_parallel_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int,
    max_in_flight: int,
    thread_name_prefix: str = "cache-prep",
) -> Iterator[R]:
    """Parallel map with bounded memory and deterministic output order."""
    workers = int(max_workers)
    in_flight = int(max_in_flight)
    if workers <= 0:
        raise ValueError("max_workers must be positive")
    if in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    in_flight = max(in_flight, workers)

    if workers == 1:
        for item in items:
            yield fn(item)
        return

    iterator = iter(items)
    pending: deque[Future[R]] = deque()
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix=str(thread_name_prefix),
    ) as pool:
        exhausted = False
        while True:
            while not exhausted and len(pending) < in_flight:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(pool.submit(fn, item))
            if not pending:
                break
            yield pending.popleft().result()
