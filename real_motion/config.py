from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class GridConfig:
    latent_hw: Tuple[int, int] = (50, 50)
    window_hw: Tuple[int, int] = (20, 20)
    max_windows: int = 8

@dataclass(frozen=True)
class FlowConfig:
    rescale_factor: float = 10.0
    sample_steps: int = 10
    alpha_shift: float = 3.0
    use_cfg: bool = False
    cfg_scale: float = 1.0

@dataclass(frozen=True)
class PriorConfig:
    use_static: bool = True
    use_kta: bool = True
    use_history_summary: bool = False
    zero_init: bool = True

@dataclass(frozen=True)
class MovingMetricContract:
    protocol: str = "interval_displacement_v2"
    speed_threshold_mps: float = 0.5
    box_margin_m: float = 0.5
    report_horizons_s: Tuple[float, ...] = (1.0, 2.0, 3.0)
