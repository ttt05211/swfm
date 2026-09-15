#!/usr/bin/env python3
"""Hard A1 evaluation for paired V17 control / V18-SE2 checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from real_motion.geometry import quaternion_yaw
from real_motion.local_st_world_model_v17 import (
    LocalSpatialTemporalWorldModelV17,
    config_from_mapping_v17,
)
from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2,
    SE2_CACHE_VERSION,
    SE2_TARGET_CONTRACT,
    YAW_ENABLED_CLASS_IDS,
    wrap_angle_np,
)
from real_motion.metrics.moving_miou_v2 import (
    Box3D,
    DYNAMIC_CLASS_IDS,
    MovingMIoUV2MultiHorizon,
    moving_support_from_world_motion,
)
from real_motion.metrics.occupancy_iou import OccupancyIoUMultiHorizon
from real_motion.msp_wm_cache import MSPWorldModelCacheDataset
from real_motion.motion_transport import FUTURE_FRAMES
from real_motion.nuscenes_adapter import NuScenesWindowSource, category_to_dynamic_class
from real_motion.rigid_transport import (
    compose_component_replacements_in_input_order,
    rasterize_rigid_component,
)
from real_motion.runtime_config import (
    add_config_args,
    load_runtime_config,
    make_prepare_config,
)
from real_motion.strong_w2det import StrongW2DetConfig, extract_instances, match_instances
from tools.real_motion import eval_p0_f9_frozen_sparse_occfm as safe
from tools.real_motion.eval_p0_f9_v17_local_stwm import (
    t0_xy_to_world_preserve_source_z,
    window_from_record,
)
from tools.real_motion.train_p0_f9_v18_se2_pair import PROTOCOL

EVAL_PROTOCOL = "p0_f9_v18_se2_hard_a1_eval_v1"
TURN_BINS = ("straight", "mild", "strong")


def load_cache(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("version") != SE2_CACHE_VERSION:
        raise RuntimeError(f"expected {SE2_CACHE_VERSION}")
    meta = obj.get("metadata") or {}
    if meta.get("se2_target_contract") != SE2_TARGET_CONTRACT:
        raise RuntimeError("SE2 target contract mismatch")
    records = obj.get("records") or []
    if not records:
        raise RuntimeError("SE2 cache has no records")
    return meta, records


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("protocol") != PROTOCOL:
        raise RuntimeError(f"checkpoint protocol mismatch: {ck.get('protocol')}")
    arm = str(ck.get("arm"))
    cfg = config_from_mapping_v17(ck.get("model_config"))
    if arm == "C":
        model = LocalSpatialTemporalWorldModelV17(cfg).to(device)
    elif arm == "Y":