import numpy as np
import torch

from real_motion.geometry import OccupancyGrid
from real_motion.local_st_world_model_v17 import (
    FRAME_MOTION_DIM,
    LocalSTWMV17Config,
)
from real_motion.local_st_world_model_v18_se2 import (
    LocalSpatialTemporalWorldModelV18SE2,
)
from real_motion.motion_transport import FEATURE_DIM, FUTURE_FRAMES, HISTORY_FRAMES
from real_motion.strong_w2det import StrongW2DetConfig
from real_motion.v19_innovation import (
    ResidualInnovationHead,
    decode_innovation,
    protected_add_only_torch,
)
from real_motion.v19_memory_adapter import (
    MemoryAdaptedV18SE2,
    load_clean_e14_into_memory_model,
)
from real_motion.v19_scene_memory import (
    PROVENANCE_OBSERVED_HISTORY,
    PROVENANCE_PERSISTENT_PREDICTION,
    SourceTrack,
    StaticWorldMemory,
    build_dynamic_source_memory,
    protected_add_only,
    render_static_history_mosaic,
)
from real_motion.v19_source_reconciliation import (
    SourceReconciliationConfig,
    effective_memory_confidence,
    reconcile_detected_sources,
    select_memory_only_tracks,
)


def _tiny_cfg():
    return LocalSTWMV17Config(
        d_model=16,
        semantic_dim=8,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=20,
        history_frames=HISTORY_FRAMES,
        future_frames=FUTURE_FRAMES,
        use_representation=True,
    )


def _random_v18_inputs(n=2):
    features = torch.randn(n, FEATURE_DIM)
    tube = torch.randint(0, 18, (n, HISTORY_FRAMES, 20, 20))
    kta = torch.randn(n, FUTURE_FRAMES, 2)
    fm = torch.randn(n, HISTORY_FRAMES, FRAME_MOTION_DIM)
    source_mask = torch.randint(0, 2, tube.shape)
    return features, tube, kta, fm, source_mask


def test_memory_adapter_current_source_is_exact_v18():
    torch.manual_seed(7)
    cfg = _tiny_cfg()
    base = LocalSpatialTemporalWorldModelV18SE2(cfg).eval()
    mem = MemoryAdaptedV18SE2(cfg).eval()
    load_clean_e14_into_memory_model(mem, base.state_dict())
    mem.freeze_clean_core()

    inputs = _random_v18_inputs()
    status = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        a = base(*inputs)
        b = mem(*inputs, history_status=status)
    assert torch.equal(a["residual_xy_m"], b["residual_xy_m"])
    assert torch.equal(a["existence_logits"], b["existence_logits"])
    assert torch.equal(a["yaw_delta_rad"], b["yaw_delta_rad"])
    assert torch.equal(b["memory_gate"], torch.zeros(2))

    trainable = mem.memory_parameter_names()
    assert trainable
    assert all(
        name.startswith(
            (
                "status_proj.",
                "memory_xy_delta.",
                "memory_yaw_delta.",
                "survival_head.",
            )
        )
        for name in trainable
    )


def test_memory_adapter_only_memory_source_has_gate():
    cfg = _tiny_cfg()
    model = MemoryAdaptedV18SE2(cfg).eval()
    inputs = _random_v18_inputs()
    status = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.3, 0.0]],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        out = model(*inputs, history_status=status)
    assert torch.equal(out["memory_gate"], torch.tensor([0.0, 1.0]))


def test_source_memory_keeps_current_order_and_appends_dormant():
    grid = OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(16, 16, 4),
    )
    free = 17
    hist = np.full((HISTORY_FRAMES, *grid.shape_hwd), free, dtype=np.uint8)
    # car: present all six frames and moving +x
    for t in range(HISTORY_FRAMES):
        hist[t, 2 + t, 2:4, 0:2] = 4
    # bus: present at t=-1.0s, absent at the final two frames -> dormant
    for t in range(HISTORY_FRAMES - 2):
        hist[t, 10, 8:10, 0:2] = 3
    poses = [np.eye(4, dtype=np.float64) for _ in range(HISTORY_FRAMES)]
    cfg = StrongW2DetConfig(free_label=free, min_component_voxels=2)

    tracks, comps = build_dynamic_source_memory(
        hist,
        poses,
        grid=grid,
        strong_cfg=cfg,
        frame_dt_s=0.5,
        max_missing_s=1.5,
    )
    assert len(comps[-1]) == 1
    assert tracks[0].observed_at_anchor
    assert tracks[0].class_id == 4
    assert tracks[0].current_component_index == 0
    assert len(tracks) >= 2
    assert any(
        (not tr.observed_at_anchor)
        and tr.provenance == PROVENANCE_OBSERVED_HISTORY
        and tr.class_id == 3
        for tr in tracks[1:]
    )



def _memory_track(
    track_id,
    class_id,
    center_xy,
    *,
    real_age_s=3.0,
    confidence=1.0,
):
    centers = np.zeros((HISTORY_FRAMES, 3), dtype=np.float64)
    centers[:, 0] = float(center_xy[0])
    centers[:, 1] = float(center_xy[1])
    valid = np.ones(HISTORY_FRAMES, dtype=bool)
    return SourceTrack(
        track_id=int(track_id),
        class_id=int(class_id),
        canonical_xyz_local=np.asarray(
            [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.0]],
            dtype=np.float64,
        ),
        centers_world=centers,
        valid_history=valid,
        velocity_world=np.zeros(3, dtype=np.float64),
        last_observed_frame=HISTORY_FRAMES - 1,
        confidence=float(confidence),
        provenance=PROVENANCE_PERSISTENT_PREDICTION,
        last_component_voxel_count=2,
        last_real_observation_age_s_override=float(real_age_s),
        detected_at_anchor_override=False,
    )


def test_source_reconciliation_preserves_detection_order_and_only_recovers_unmatched():
    memory = [
        _memory_track(10, 4, (5.0, 5.0)),
        _memory_track(11, 3, (10.0, 10.0)),
        _memory_track(12, 4, (20.0, 20.0)),
    ]
    detected = [
        {"class_id": 4, "centroid_world": np.asarray([5.5, 5.0, 0.0])},
        {"class_id": 6, "centroid_world": np.asarray([2.0, 2.0, 0.0])},
        {"class_id": 3, "centroid_world": np.asarray([10.2, 10.0, 0.0])},
    ]
    cfg = SourceReconciliationConfig(
        max_center_distance_m=4.0,
        max_memory_age_s=6.0,
        confidence_tau_s=3.0,
        min_memory_confidence=0.15,
    )
    rec = reconcile_detected_sources(
        detected,
        memory,
        frame_dt_s=0.5,
        config=cfg,
    )
    assert [m.detected_index for m in rec.matches] == [0, 2]
    assert [m.memory_index for m in rec.matches] == [0, 1]
    assert rec.unmatched_detected == (1,)
    assert rec.unmatched_memory == (2,)

    sel = select_memory_only_tracks(
        memory,
        rec,
        frame_dt_s=0.5,
        config=cfg,
    )
    assert sel.memory_indices == (2,)
    assert len(sel.tracks) == 1
    assert sel.tracks[0].track_id == 12
    assert not sel.tracks[0].detected_at_anchor
    assert sel.tracks[0].confidence < 1.0


def test_source_reconciliation_never_matches_wrong_class_even_when_closer():
    memory = [_memory_track(1, 4, (0.0, 0.0))]
    detected = [
        {"class_id": 3, "centroid_world": np.asarray([0.0, 0.0, 0.0])}
    ]
    rec = reconcile_detected_sources(
        detected,
        memory,
        frame_dt_s=0.5,
        config=SourceReconciliationConfig(max_center_distance_m=4.0),
    )
    assert rec.matches == ()
    assert rec.unmatched_detected == (0,)
    assert rec.unmatched_memory == (0,)


def test_memory_recovery_age_and_confidence_gates_are_causal():
    memory = [
        _memory_track(1, 4, (0.0, 0.0), real_age_s=3.0, confidence=1.0),
        _memory_track(2, 4, (10.0, 0.0), real_age_s=7.0, confidence=1.0),
    ]
    rec = reconcile_detected_sources(
        [],
        memory,
        frame_dt_s=0.5,
        config=SourceReconciliationConfig(max_center_distance_m=4.0),
    )
    cfg = SourceReconciliationConfig(
        max_center_distance_m=4.0,
        max_memory_age_s=6.0,
        confidence_tau_s=3.0,
        min_memory_confidence=0.30,
    )
    sel = select_memory_only_tracks(
        memory,
        rec,
        frame_dt_s=0.5,
        config=cfg,
    )
    # age=3, tau=3 -> exp(-1)=0.3679: retained; age=7: rejected.
    assert sel.memory_indices == (0,)
    assert sel.dropped_by_age == (1,)
    assert np.isclose(
        effective_memory_confidence(
            memory[0], frame_dt_s=0.5, confidence_tau_s=3.0
        ),
        np.exp(-1.0),
    )

def test_static_memory_observed_free_clears_but_dynamic_does_not():
    grid = OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(4, 4, 2),
    )
    free = 17
    pose = np.eye(4, dtype=np.float64)
    mem = StaticWorldMemory(grid.voxel_size)

    sem = np.full(grid.shape_hwd, free, dtype=np.uint8)
    obs = np.zeros(grid.shape_hwd, dtype=bool)
    sem[1, 1, 0] = 11
    obs[1, 1, 0] = True
    mem.update(sem, obs, pose, grid=grid, free_label=free, frame_index=0)
    assert mem.render(pose, grid=grid, free_label=free)[1, 1, 0] == 11

    # A dynamic observation at the same world cell must not erase static memory.
    sem2 = np.full(grid.shape_hwd, free, dtype=np.uint8)
    obs2 = np.zeros(grid.shape_hwd, dtype=bool)
    sem2[1, 1, 0] = 4
    obs2[1, 1, 0] = True
    mem.update(sem2, obs2, pose, grid=grid, free_label=free, frame_index=1)
    assert mem.render(pose, grid=grid, free_label=free)[1, 1, 0] == 11

    # A genuinely observed free cell invalidates the stale static voxel.
    sem3 = np.full(grid.shape_hwd, free, dtype=np.uint8)
    obs3 = np.zeros(grid.shape_hwd, dtype=bool)
    obs3[1, 1, 0] = True
    mem.update(sem3, obs3, pose, grid=grid, free_label=free, frame_index=2)
    assert mem.render(pose, grid=grid, free_label=free)[1, 1, 0] == free



def test_fast_static_mosaic_recent_free_wins_and_dynamic_does_not_clear():
    grid = OccupancyGrid(
        x_min=0.0,
        y_min=0.0,
        z_min=0.0,
        voxel_size=(1.0, 1.0, 1.0),
        shape_hwd=(4, 4, 2),
    )
    free = 17
    hist = np.full((HISTORY_FRAMES, *grid.shape_hwd), free, dtype=np.uint8)
    obs = np.zeros_like(hist, dtype=bool)
    poses = [np.eye(4, dtype=np.float64) for _ in range(HISTORY_FRAMES)]

    # Old static evidence.
    hist[0, 1, 1, 0] = 11
    obs[0, 1, 1, 0] = True
    # Dynamic occluder does not clear the map.
    hist[1, 1, 1, 0] = 4
    obs[1, 1, 1, 0] = True
    out = render_static_history_mosaic(
        hist, obs, poses, poses[-1], grid=grid, free_label=free
    )
    assert out[1, 1, 0] == 11

    # Newer genuinely observed free evidence does clear it.
    obs[-1, 1, 1, 0] = True
    out = render_static_history_mosaic(
        hist, obs, poses, poses[-1], grid=grid, free_label=free
    )
    assert out[1, 1, 0] == free

def test_protected_add_only_never_overwrites_base():
    free = 17
    base = np.full((3, 3, 2), free, dtype=np.uint8)
    base[1, 1, 0] = 4
    proposal = np.full_like(base, free)
    proposal[1, 1, 0] = 3
    proposal[2, 2, 1] = 11
    out = protected_add_only(base, proposal, free_label=free)
    assert out[1, 1, 0] == 4
    assert out[2, 2, 1] == 11


def test_innovation_head_shapes_and_torch_protection():
    B, Fh, T, H, W, Z = 2, 6, 6, 20, 20, 16
    model = ResidualInnovationHead(
        future_frames=Fh,
        history_frames=T,
        semantic_dim=4,
        hidden_dim=8,
        num_semantic_classes=17,
        vertical_bins=Z,
    ).eval()
    sem = torch.randint(0, 18, (B, Fh, T, H, W))
    geo = torch.rand(B, Fh, T, 4, H, W)
    explained = torch.zeros(B, Fh, 1, H, W)
    with torch.inference_mode():
        out = model(sem, geo, explained)
    assert out["add_presence_logits"].shape == (B, Fh, H, W)
    assert out["semantic_logits"].shape == (B, Fh, 17, H, W)
    assert out["vertical_occupancy_logits"].shape == (B, Fh, Z, H, W)

    proposal = decode_innovation(
        out, free_label=17, add_threshold=0.5, vertical_threshold=0.5
    )
    assert proposal.shape == (B, Fh, H, W, Z)

    base = torch.full_like(proposal, 17)
    base[..., 0, 0, 0] = 4
    proposal[..., 0, 0, 0] = 3
    proposal[..., 1, 1, 0] = 11
    merged = protected_add_only_torch(base, proposal, free_label=17)
    assert torch.all(merged[..., 0, 0, 0] == 4)
    assert torch.all(merged[..., 1, 1, 0] == 11)
