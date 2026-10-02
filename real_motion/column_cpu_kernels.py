"""Exact NumPy kernels on caller-owned float64 coordinate workspaces."""
import numpy as np


def inside_xyz(indices, shape):
    # Avoid constructing two N*3 bool arrays and a short-axis reduction.
    x, y, z = indices.T
    return ((x >= 0) & (x < shape[0]) & (y >= 0) & (y < shape[1])
            & (z >= 0) & (z < shape[2]))


def metric_indices_inplace(points, origin, step, shape):
    """Consume a fresh transform output; do NOT pass shared history points.

    Keep float64 subtraction, division and floor in the reference order. This
    is not a fused/float32 approximation and does not reassociate transforms.
    """
    np.subtract(points, origin, out=points)
    np.divide(points, step, out=points)
    np.floor(points, out=points)
    indices = points.astype(np.int64)
    return indices, inside_xyz(indices, shape)
