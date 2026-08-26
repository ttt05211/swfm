"""SWFM command-line tools package.

This file intentionally makes the repository-local ``tools`` directory a
regular Python package. Several entrypoints reuse helpers from sibling SWFM
tools; without this marker, the pinned upstream OccFM ``tools`` package can
shadow this directory when ``upstream_occfm`` is also on ``sys.path``.
"""
