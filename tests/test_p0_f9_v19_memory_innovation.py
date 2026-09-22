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
    SourceTrack,
    StaticWorldMemory,
    build_dynamic_source_memory,
    protected_add_only,
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
