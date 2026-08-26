#!/usr/bin/env python3
"""Build sharded P0/raw prepared windows from nuScenes + Occ3D labels."""
import argparse, sys
from pathlib import Path
from dataclasses import asdict
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from real_motion.nuscenes_adapter import NuScenesWindowSource
from real_motion.prepared import PrepareConfig, prepare_nuscenes_window, save_prepared_shards
from real_motion.kta import KTAConfig
from real_motion.motion import PersistenceMotionConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True)
    p.add_argument("--info-pkl", required=True, help="official temporal info pkl")
    p.add_argument("--output", required=True)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--shard-size", type=int, default=16)
    p.add_argument("--static-persistence", type=float, default=0.80)
    p.add_argument("--moving-persistence", type=float, default=0.50)
    p.add_argument("--component-tracks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--component-max-step-m", type=float, default=4.0)
    p.add_argument("--component-moving-speed-mps", type=float, default=0.5)
    p.add_argument("--component-static-speed-mps", type=float, default=0.2)
    p.add_argument("--component-min-track-frames", type=int, default=2)
    p.add_argument("--kta-max-match-m", type=float, default=6.0)
    p.add_argument("--tube-radii", default="1,2,3,4,5,6")
    a = p.parse_args()

    radii = tuple(int(x) for x in a.tube_radii.split(","))
    if len(radii) != 6:
        raise ValueError("OccFM 3s protocol expects six 0.5s tube radii")
    cfg = PrepareConfig(
        tube_radii=radii,
        motion=PersistenceMotionConfig(
            static_min_persistence=a.static_persistence,
            moving_max_persistence=a.moving_persistence,
            use_component_tracks=a.component_tracks,
            component_max_step_m=a.component_max_step_m,
            moving_speed_mps=a.component_moving_speed_mps,
            static_speed_mps=a.component_static_speed_mps,
            min_track_frames=a.component_min_track_frames,
        ),
        kta=KTAConfig(max_match_distance_m=a.kta_max_match_m),
    )
    source = NuScenesWindowSource(a.dataroot, info_pkl=a.info_pkl, verbose=False)

    def generator():
        for i, window in enumerate(source.iter_windows(
                history=cfg.history_frames, future=cfg.future_frames,
                stride=a.stride, max_windows=a.max_windows)):
            sample = prepare_nuscenes_window(source, window, cfg)
            if i % 25 == 0:
                print("prepared", i, sample["sample_id"])
            yield sample

    meta = {
        "dataroot": str(Path(a.dataroot).resolve()),
        "info_pkl": str(Path(a.info_pkl).resolve()),
        "history_frames": 6, "future_frames": 6, "frame_dt_s": 0.5,
        "tube_radii": list(radii),
        "motion_config": asdict(cfg.motion),
        "kta_config": asdict(cfg.kta),
        "grid": asdict(cfg.grid),
        "causal": True,
    }
    index = save_prepared_shards(a.output, generator(), a.shard_size, meta)
    print("saved", index["num_samples"], "prepared windows to", a.output)


if __name__ == "__main__":
    main()
