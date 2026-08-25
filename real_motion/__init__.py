"""Real-Motion Sparse Occupancy World Model extension for OccFM."""

from .support import MotionTubeConfig, build_motion_tube, downsample_support
from .windows import WindowPlan, WindowPlanner, crop_windows, scatter_windows, window_coverage
from .composition import static_protected_compose

__all__ = [
    "MotionTubeConfig", "build_motion_tube", "downsample_support",
    "WindowPlan", "WindowPlanner", "crop_windows", "scatter_windows", "window_coverage",
    "static_protected_compose",
]
