"""Read-only/exact execution candidates for Shared-History training.

This module intentionally keeps the scientific model, REMOVE path, six-horizon
CE+Lovasz objective, motion/ego supervision and optimizer semantics unchanged.
It only changes execution grouping:

* sequential H3/H6: fewer renderer/backward groups, same six horizons;
* packed B4: one shared-history encoder/STWM/repair forward across the existing
  global window batch, followed by the same per-scene renderer and one shared
  network backward.

Nothing here is promoted automatically to the scientific trainer.
"""
from collections import defaultdict
import time
import numpy as np
import torch
from torch.nn import functional as F

from tools.shared_history_experiments.model import sample_points, SharedJoint
from tools.shared_history_experiments import engine as reference
from tools.joint_occ_experiments.data import prepare_projection
from tools.joint_occ_experiments.model import source_auxiliary, ego_geometry_loss
from tools.joint_occ_experiments.soft_geometry import (
    source_targets, effective_yaw, continuous_context, grid_arrays, move,
)
from tools.joint_occ_experiments.fast_autograd import RenderGradientBridge, compose_sparse
from tools.joint_occ_experiments.batching.geometry import GroupedScene, grouped_loss, grouped_splat


class WideGroupedScene(GroupedScene):
    """Same renderer as GroupedScene, but allow all six future slots at once."""

    def render_group(self, horizons, poses, targets, yaws, mapped, evidence, plan, actions, *, edits=True):
        if not horizons or len(horizons) > 6 or len(set(horizons)) != len(horizons):
            raise ValueError("one to six DISTINCT future horizons required")
        if any(h not in range(6) for h in horizons):
            raise ValueError("future horizon out of range")
        device = poses.device
        hs = torch.as_tensor(horizons, device=device)
        volume = int(np.prod(self.shape))
        count = len(horizons)

        selected = poses[hs]
        relative = torch.linalg.inv(self.current_pose)[None] @ selected
        xyz = (
            torch.einsum("gij,nj->gni", relative[:, :3, :3], self.centers.reshape(-1, 3))
            + relative[:, None, :3, 3]
        )
        origin, step, _ = grid_arrays(self.grid)
        q = (xyz - xyz.new_tensor(origin)) / xyz.new_tensor(step)
        sampling = (q / q.new_tensor(self.shape) * 2 - 1)[..., [2, 1, 0]].float().reshape(
            count, *self.shape, 3
        )
        background = F.grid_sample(
            self.static.expand(count, -1, -1, -1, -1),
            sampling,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(count, 18, volume)
        missing = (1 - background.sum(1)).clamp(0, 1)
        background = background.scatter_add(
            1,
            torch.stack([self.backgrounds[h] for h in horizons])[:, None],
            missing[:, None],
        )
        background = background.permute(1, 0, 2).reshape(18, count * volume)

        owners = torch.stack([self.owners[h] for h in horizons]).flatten()
        baseline = torch.stack([self.baselines[h] for h in horizons]).flatten()
        fallback = torch.stack([self.fallbacks[h] for h in horizons]).flatten()

        operations, alphas = [], []
        actor = np.asarray(evidence.actor)
        legal = np.asarray(plan.legal)

        def edit(ids, action):
            rows, slots, times = [], [], []
            for slot, h in enumerate(horizons):
                at = ids[legal[ids, h, action]]
                if len(at):
                    rows.append(torch.as_tensor(at, device=device))
                    slots.append(torch.full((len(at),), slot, device=device, dtype=torch.long))
                    times.append(torch.full((len(at),), h, device=device, dtype=torch.long))
            if not rows:
                return None
            rows, slots, times = torch.cat(rows), torch.cat(slots), torch.cat(times)
            return grouped_splat(
                mapped[rows, times], actions[rows, times, action], slots, self.grid
            )

        def append(kind, ids, alpha, cls, destination=None):
            operations.append((kind, ids, cls, destination))
            alphas.append(alpha)

        # REMOVE is deliberately preserved in this speed work.
        if edits:
            for cls in (11, 13):
                result = edit(np.flatnonzero((actor < 0) & (evidence.classes == cls)), 1)
                if result is not None:
                    ids, alpha = result
                    alpha = alpha * ((owners[ids] < 0) & (baseline[ids] == cls))
                    append("remove", ids, alpha, cls, torch.full_like(ids, 17))

        inverse = torch.linalg.inv(selected)
        for i, cls in enumerate(self.classes):
            points = move(
                self.source_points[i][None],
                self.source_centers[i],
                targets[i, hs, None],
                yaws[i, hs, None],
            )
            points = (
                torch.einsum("gij,gnj->gni", inverse[:, :3, :3], points)
                + inverse[:, None, :3, 3]
            )
            slots = (
                torch.arange(count, device=device)[:, None]
                .expand(-1, len(self.source_points[i]))
                .flatten()
            )
            points = points.reshape(-1, 3)
            ids, alpha = grouped_splat(
                points, points.new_ones(len(points)).float(), slots, self.grid, "sum"
            )
            append("overlay", ids, alpha, cls)

        if edits:
            for i, cls in enumerate(self.classes):
                result = edit(np.flatnonzero(actor == i), 1)
                if result is not None:
                    ids, alpha = result
                    alpha = alpha * ((owners[ids] == i) & (baseline[ids] == cls))
                    append("remove", ids, alpha, cls, fallback[ids])
            for actor_id in np.unique(actor):
                for cls in np.unique(evidence.classes[actor == actor_id]):
                    result = edit(
                        np.flatnonzero((actor == actor_id) & (evidence.classes == cls)), 0
                    )
                    if result is not None:
                        ids, alpha = result
                        append("overlay", ids, alpha * (baseline[ids] == 17), int(cls))

        return compose_sparse(background, operations, alphas).reshape(
            18, count, volume
        ).permute(1, 0, 2)


class PackedSceneRenderer:
    """Exact scene x horizon slot batching for the existing generic renderer.

    Each scene keeps its own source order, REMOVE owner/fallback semantics and
    ADD actor/class order.  Scene slots are disjoint in the flattened volume,
    so operations from different scenes can share grid_sample/splat/loss calls
    without changing same-scene composition order.
    """

    def __init__(self, forwards, grid, device):
        if not forwards:
            raise ValueError("packed renderer requires at least one scene")
        self.scenes = [
            WideGroupedScene(fw["prep"], grid, device) for fw in forwards
        ]
        self.grid = grid
        self.device = device
        self.shape = self.scenes[0].shape
        if any(scene.shape != self.shape for scene in self.scenes):
            raise ValueError("packed scenes must share occupancy grid shape")
        self.volume = int(np.prod(self.shape))
        self.centers = self.scenes[0].centers
        self.static = torch.cat([scene.static for scene in self.scenes], 0)
        self.current_pose = torch.stack([scene.current_pose for scene in self.scenes])

    def render_group(self, horizons, leaves, forwards, *, edits=True):
        if len(leaves) != len(self.scenes) or len(forwards) != len(self.scenes):
            raise ValueError("packed renderer scene count differs")
        if not horizons or len(horizons) > 6 or len(set(horizons)) != len(horizons):
            raise ValueError("one to six DISTINCT future horizons required")
        if any(h not in range(6) for h in horizons):
            raise ValueError("future horizon out of range")

        device = self.device
        batch, count, volume = len(self.scenes), len(horizons), self.volume
        hs = torch.as_tensor(horizons, device=device)

        # One dense background warp for B x H independent scene-horizon slots.
        poses = torch.stack([leaf["poses"] for leaf in leaves])
        selected = poses[:, hs]
        relative = torch.linalg.inv(self.current_pose)[:, None] @ selected
        xyz = (
            torch.einsum(
                "bgij,nj->bgni",
                relative[..., :3, :3],
                self.centers.reshape(-1, 3),
            )
            + relative[..., None, :3, 3]
        )
        origin, step, _ = grid_arrays(self.grid)
        q = (xyz - xyz.new_tensor(origin)) / xyz.new_tensor(step)
        sampling = (
            (q / q.new_tensor(self.shape) * 2 - 1)[..., [2, 1, 0]]
            .float()
            .reshape(batch * count, *self.shape, 3)
        )
        static = (
            self.static[:, None]
            .expand(batch, count, *self.static.shape[1:])
            .reshape(batch * count, *self.static.shape[1:])
        )
        background = F.grid_sample(
            static,
            sampling,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).reshape(batch * count, 18, volume)
        missing = (1 - background.sum(1)).clamp(0, 1)
        backgrounds = torch.stack(
            [
                scene.backgrounds[h]
                for scene in self.scenes
                for h in horizons
            ]
        )
        background = background.scatter_add(
            1, backgrounds[:, None], missing[:, None]
        )
        background = background.permute(1, 0, 2).reshape(
            18, batch * count * volume
        )
        owners = torch.stack(
            [scene.owners[h] for scene in self.scenes for h in horizons]
        ).flatten()
        baseline = torch.stack(
            [scene.baselines[h] for scene in self.scenes for h in horizons]
        ).flatten()
        fallback = torch.stack(
            [scene.fallbacks[h] for scene in self.scenes for h in horizons]
        ).flatten()

        actors = [np.asarray(fw["evidence"].actor) for fw in forwards]
        classes = [np.asarray(fw["evidence"].classes) for fw in forwards]
        legals = [np.asarray(fw["plan"].legal) for fw in forwards]
        operations, alphas = [], []

        def edit_many(items, action):
            points, values, slots = [], [], []
            for scene_index, ids in items:
                if not len(ids):
                    continue
                legal = legals[scene_index]
                mapped = leaves[scene_index]["mapped"]
                actions = leaves[scene_index]["actions"]
                for local_slot, h in enumerate(horizons):
                    at_np = ids[legal[ids, h, action]]
                    if not len(at_np):
                        continue
                    at = torch.as_tensor(at_np, device=device)
                    points.append(mapped[at, h])
                    values.append(actions[at, h, action])
                    slots.append(
                        torch.full(
                            (len(at),),
                            scene_index * count + local_slot,
                            device=device,
                            dtype=torch.long,
                        )
                    )
            if not points:
                return None
            return grouped_splat(
                torch.cat(points),
                torch.cat(values),
                torch.cat(slots),
                self.grid,
            )

        def append(kind, ids, alpha, cls, destination=None):
            operations.append((kind, ids, cls, destination))
            alphas.append(alpha)

        # Preserve the full generic REMOVE path.
        if edits:
            for cls in (11, 13):
                result = edit_many(
                    [
                        (
                            scene_index,
                            np.flatnonzero(
                                (actors[scene_index] < 0)
                                & (classes[scene_index] == cls)
                            ),
                        )
                        for scene_index in range(batch)
                    ],
                    1,
                )
                if result is not None:
                    ids, alpha = result
                    alpha = alpha * ((owners[ids] < 0) & (baseline[ids] == cls))
                    append("remove", ids, alpha, cls, torch.full_like(ids, 17))

        # Source overlays are ordered by local source rank exactly as in every
        # independent scene.  At one rank, different scenes occupy disjoint slots.
        max_sources = max(len(scene.classes) for scene in self.scenes)
        for source_index in range(max_sources):
            for cls in range(18):
                point_parts, slot_parts = [], []
                for scene_index, scene in enumerate(self.scenes):
                    if (
                        source_index >= len(scene.classes)
                        or scene.classes[source_index] != cls
                    ):
                        continue
                    leaf = leaves[scene_index]
                    selected_pose = leaf["poses"][hs]
                    inverse = torch.linalg.inv(selected_pose)
                    source = move(
                        scene.source_points[source_index][None],
                        scene.source_centers[source_index],
                        leaf["targets"][source_index, hs, None],
                        leaf["yaws"][source_index, hs, None],
                    )
                    source = (
                        torch.einsum(
                            "gij,gnj->gni", inverse[:, :3, :3], source
                        )
                        + inverse[:, None, :3, 3]
                    )
                    point_parts.append(source.reshape(-1, 3))
                    slot_parts.append(
                        (
                            scene_index * count
                            + torch.arange(count, device=device)[:, None]
                        )
                        .expand(-1, len(scene.source_points[source_index]))
                        .reshape(-1)
                    )
                if point_parts:
                    points = torch.cat(point_parts)
                    slots = torch.cat(slot_parts)
                    ids, alpha = grouped_splat(
                        points,
                        points.new_ones(len(points)).float(),
                        slots,
                        self.grid,
                        "sum",
                    )
                    append("overlay", ids, alpha, cls)

        if edits:
            # Dynamic REMOVE keeps local actor rank/owner identity.
            for source_index in range(max_sources):
                for cls in range(18):
                    items = []
                    for scene_index, scene in enumerate(self.scenes):
                        if (
                            source_index < len(scene.classes)
                            and scene.classes[source_index] == cls
                        ):
                            items.append(
                                (
                                    scene_index,
                                    np.flatnonzero(
                                        actors[scene_index] == source_index
                                    ),
                                )
                            )
                    result = edit_many(items, 1)
                    if result is not None:
                        ids, alpha = result
                        alpha = alpha * (
                            (owners[ids] == source_index)
                            & (baseline[ids] == cls)
                        )
                        append("remove", ids, alpha, cls, fallback[ids])

            # ADD follows the original sorted actor -> sorted class order.
            for actor_id in range(-1, max_sources):
                for cls in range(18):
                    items = []
                    for scene_index in range(batch):
                        ids = np.flatnonzero(
                            (actors[scene_index] == actor_id)
                            & (classes[scene_index] == cls)
                        )
                        if len(ids):
                            items.append((scene_index, ids))
                    result = edit_many(items, 0)
                    if result is not None:
                        ids, alpha = result
                        append(
                            "overlay",
                            ids,
                            alpha * (baseline[ids] == 17),
                            cls,
                        )

        return compose_sparse(background, operations, alphas).reshape(
            18, batch * count, volume
        ).permute(1, 0, 2)


def _groups(horizon_group):
    if horizon_group not in (2, 3, 6):
        raise ValueError("horizon_group must be 2, 3 or 6")
    return [
        list(range(first, min(6, first + horizon_group)))
        for first in range(0, 6, horizon_group)
    ]


def train_arm_grouped(
    model,
    optimizer,
    provider,
    rows,
    mode,
    *,
    update,
    total_updates,
    object_weight=1.0,
    ego_weight=1.0,
    lovasz_weight=1.0,
    probe=False,
    optimized=True,
    horizon_group=3,
):
    """Reference single-window network execution with H3/H6 renderer grouping."""
    if mode != "gt" or not optimized or not rows:
        raise ValueError("GT-only fused shared-history speed candidate required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    factor = 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (update - 1) / max(1, total_updates)))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor

    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    sources = canonical = 0

    for row in rows:
        tick = time.perf_counter()
        fw = reference.forward_window(model, provider, row)
        scene = WideGroupedScene(fw["prep"], provider.pcfg.grid, provider.device)
        bridge = RenderGradientBridge(fw)
        stages["shared_encoder_motion_projection_ccr"] += time.perf_counter() - tick
        sources += len(fw["output"]["residual_xy_m"])
        canonical += len(fw["evidence"])
        labels = np.asarray(row[1]["future_gt_occ"])

        for hs in _groups(horizon_group):
            tick = time.perf_counter()
            probability = scene.render_group(
                hs, **bridge.leaves, evidence=fw["evidence"], plan=fw["plan"]
            )
            losses, ce, lovasz = grouped_loss(
                probability, labels[hs], lovasz_weight=lovasz_weight
            )
            if not bool(torch.isfinite(losses).all()):
                raise RuntimeError("nonfinite full-grid occupancy objective")
            if probe and 1 in hs and losses.requires_grad:
                tensors, names = [fw["actions"]], ["occupancy_to_actions"]
                if "field" in fw:
                    tensors += [fw["field"]]
                    names += ["occupancy_to_shared_history"]
                if len(fw["output"]["residual_xy_m"]):
                    tensors += [
                        fw["output"]["residual_xy_m"],
                        fw["output"]["yaw_delta_rad"],
                    ]
                    names += ["occupancy_to_xy", "occupancy_to_yaw"]
                for name, grad in zip(
                    names, bridge.probe(losses[hs.index(1)], tensors)
                ):
                    links[name] += 0.0 if grad is None else float(grad.detach().float().norm())
            stages["soft_render_full_ce_lovasz"] += time.perf_counter() - tick

            tick = time.perf_counter()
            if losses.requires_grad:
                (losses / (len(rows) * 6)).sum().backward()
            stages["occupancy_backward"] += time.perf_counter() - tick
            terms["CE"].extend(ce.detach().unbind())
            terms["Lovasz"].extend(lovasz.detach().unbind())
            del probability, losses, ce, lovasz

        tick = time.perf_counter()
        geometry, existence = source_auxiliary(fw["output"], row[0], fw["prep"].state)
        ego_loss = ego_geometry_loss(
            fw["ego"][None], np.asarray(fw["ego_target"])[None]
        )
        auxiliary = object_weight * (geometry + existence) + ego_weight * ego_loss
        if not bool(torch.isfinite(auxiliary)):
            raise RuntimeError("nonfinite direct geometry objective")
        bridge.backward_shared(auxiliary / len(rows))
        terms["source_geometry"].append(geometry.detach())
        terms["existence"].append(existence.detach())
        terms["ego_geometry"].append(ego_loss.detach())
        stages["shared_network_and_direct_geometry_backward"] += time.perf_counter() - tick
        del fw, scene, bridge, geometry, existence, ego_loss, auxiliary

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    tick = time.perf_counter()
    optimizer.step()
    if provider.device.type == "cuda":
        torch.cuda.synchronize(provider.device)
    stages["optimizer_finish"] += time.perf_counter() - tick

    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(
        mode=mode,
        windows=len(rows),
        sources=sources,
        canonical_points=canonical,
        seconds=time.perf_counter() - begun,
        stages_seconds=dict(stages),
        grad_norm=float(norm),
        gradient_links=dict(links),
        training_backend=f"shared_history_sequential_h{horizon_group}",
        peak_memory_mib=(
            torch.cuda.max_memory_allocated(provider.device) / 2**20
            if provider.device.type == "cuda"
            else 0.0
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    return result


def _encode_fields(model, rows, device):
    """One SharedEncoder stem call for B existing windows; parameters unchanged."""
    encoder = model.transport.history_encoder
    semantics = torch.as_tensor(
        np.stack([np.asarray(row[1]["history_occ"]) for row in rows]), device=device
    )
    observed = torch.as_tensor(
        np.stack([np.asarray(row[1]["history_observed"]) for row in rows]), device=device
    )
    if semantics.ndim != 5 or semantics.shape[1] != 4 or semantics.shape[-1] != encoder.z_bins:
        raise ValueError("packed encoder requires [B,4,X,Y,Z]")
    known = observed.bool() & (semantics >= 0) & (semantics < 18)
    labels = torch.where(known, semantics.long(), 18)
    x = torch.cat((encoder.semantic(labels), known[..., None].float()), -1)
    # [B,4,X,Y,Z,5] -> [B*4,Z*5,X,Y]
    x = x.permute(0, 1, 4, 5, 2, 3).flatten(2, 3)
    batch, frames, channels, sx, sy = x.shape
    field = encoder.stem(x.reshape(batch * frames, channels, sx, sy))
    return field.reshape(batch, frames, *field.shape[1:])


def _pack_transport(model, fields, reads, times, classes, ktas):
    transport = model.transport
    if not fields:
        raise ValueError("packed transport requires at least one field view")
    template = fields[0]
    dim, hw = transport.config.d_model, transport.config.tube_hw // 2
    local_parts, condition_parts, class_parts, kta_parts, lengths = [], [], [], [], []

    for field, read, t, cls, kta in zip(fields, reads, times, classes, ktas):
        sources = int(read["source_grid"].shape[0])
        lengths.append(sources)
        if not sources:
            continue
        if read["source_grid"].shape != (sources, 4, hw, hw, 2):
            raise ValueError("source crop physical/sample contract differs")
        values = F.grid_sample(
            field.float(),
            read["source_grid"].permute(1, 0, 2, 3, 4).reshape(4, sources * hw, hw, 2),
            padding_mode="zeros",
            align_corners=False,
        ).reshape(4, field.shape[1], sources, hw, hw).permute(2, 0, 1, 3, 4)
        valid = read["source_valid"].float()
        center = read["source_center"]
        offset = (center - center[:, -1:]) / 20
        dt = (t[1:] - t[:-1]).clamp_min(0.001) * 1.5
        segment_valid = valid[:, 1:] * valid[:, :-1]
        segment = (
            (center[:, 1:] - center[:, :-1])
            / dt[None, :, None]
            / 20
            * segment_valid[..., None]
        )
        velocity = torch.cat((segment[:, :1], segment), 1)
        condition = torch.cat(
            (
                offset * valid[..., None],
                velocity * valid[..., None],
                valid[..., None],
                t[None, :, None].expand(sources, -1, -1),
            ),
            -1,
        )
        local = torch.cat(
            (
                values * valid[:, :, None, None, None],
                read["source_mask"][:, :, None],
            ),
            2,
        )
        local_parts.append(local.reshape(sources * 4, -1, hw, hw))
        condition_parts.append(condition)
        class_parts.append(cls)
        kta_parts.append(kta)

    total = sum(lengths)
    if not total:
        empty = template.new_empty
        output = dict(
            residual_xy_m=empty((0, 6, 2)),
            yaw_delta_rad=empty((0, 6)),
            existence_logits=empty((0, 6)),
            history_source_context=empty((0, dim)),
            future_transport_queries=empty((0, 6, dim)),
        )
        return output, lengths

    local = torch.cat(local_parts, 0)
    condition = torch.cat(condition_parts, 0)
    classes = torch.cat(class_parts, 0)
    kta = torch.cat(kta_parts, 0)
    x = transport.source_adapter(local).reshape(total, 4, dim, hw, hw)
    x = (
        x
        + transport.motion_condition(condition)[:, :, :, None, None]
        + transport.class_embedding(classes)[:, None, :, None, None]
        + transport.time_embedding
        + transport.spatial_embedding
    )
    for block in transport.blocks:
        x = block(x)
    context = x.permute(0, 1, 3, 4, 2).reshape(total, -1, dim)
    q = transport.future_query.expand(total, -1, -1) + transport.kta_future_proj(
        kta.to(x.dtype) / 20
    )
    for block in transport.decoder:
        q = block(q, context)
    return dict(
        residual_xy_m=transport.residual_head(q),
        yaw_delta_rad=transport.yaw_head(q)[..., 0],
        existence_logits=transport.existence_head(q)[..., 0],
        history_source_context=context.mean(1),
        future_transport_queries=q,
    ), lengths


def _split_sources(output, lengths):
    result, start = [], 0
    for length in lengths:
        stop = start + length
        result.append({key: value[start:stop] for key, value in output.items()})
        start = stop
    return result


def _encode_repair_packed(model, fields, reads, times, evidences):
    if not fields:
        raise ValueError("packed repair requires at least one field view")
    template = fields[0]
    columns = model.columns
    values_parts, facts_parts, valid_parts = [], [], []
    actor_parts, class_parts, surface_parts, point_lengths = [], [], [], []

    for field, read, t, evidence in zip(fields, reads, times, evidences):
        values = sample_points(field, read["point_grid"])
        valid = read["point_valid"].bool()
        n = len(values)
        point_lengths.append(n)
        if not n:
            continue
        facts = torch.stack(
            (
                read["point_z"],
                t[None].expand(n, -1),
                valid.float(),
                torch.as_tensor(evidence.presence, device=field.device).float(),
            ),
            -1,
        )
        values_parts.append(values)
        facts_parts.append(facts)
        valid_parts.append(valid)
        actor_parts.append(torch.as_tensor(evidence.actor, device=field.device).long())
        class_parts.append(torch.as_tensor(evidence.classes, device=field.device).long())
        surface_parts.append(
            torch.as_tensor(evidence.features[:, 27:], device=field.device)
        )

    total = sum(point_lengths)
    if not total:
        return template.new_empty((0, columns.width)), [], point_lengths

    values = torch.cat(values_parts, 0)
    facts = torch.cat(facts_parts, 0)
    valid = torch.cat(valid_parts, 0)
    actor = torch.cat(actor_parts, 0)
    classes = torch.cat(class_parts, 0)
    surface = torch.cat(surface_parts, 0)

    x = columns.point_adapter(torch.cat((values, facts), -1))
    score = columns.time_score(x)[..., 0].float().masked_fill(~valid, -1e4)
    weight = score.softmax(1) * valid.float()
    weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-8)
    x = (x * weight[..., None]).sum(1)
    encoded = (
        x
        + columns.surface(surface) * (actor < 0)[:, None]
        + columns.role((actor >= 0).long())
        + columns.class_projection(columns.semantic(classes))
    )
    return encoded, actor_parts, point_lengths


def forward_batch(model, provider, rows):
    """Pack the trainable Shared-History network; keep per-scene geometry exact."""
    if not isinstance(model, SharedJoint) or not 1 <= len(rows) <= 4:
        raise ValueError("packed SharedJoint requires 1..4 existing global-batch windows")
    device = provider.device
    cast = lambda value: torch.as_tensor(np.asarray(value), device=device)

    reads, times, classes, ktas = [], [], [], []
    for record, raw, ego_inputs, history in rows:
        reads.append(
            {key: torch.as_tensor(value, device=device) for key, value in history[0]["shared_reads"].items()}
        )
        times.append(torch.as_tensor(ego_inputs[0][:, -1], device=device))
        classes.append(cast(record["source_class_id"]).long())
        ktas.append(cast(record["kta_displacement_xy_m"]).float())

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        fields = _encode_fields(model, rows, device)
        # IMPORTANT: create the per-scene views ONCE and reuse these exact
        # Tensor objects everywhere.  Re-indexing fields[index] later creates
        # a different view object; autograd.grad(view_not_consumed, ...) then
        # reports None even though encoder parameters receive real gradients.
        field_views = list(fields.unbind(0))
        packed_output, source_lengths = _pack_transport(
            model, field_views, reads, times, classes, ktas
        )
        ego_history = cast(np.stack([row[2][0] for row in rows]))
        commands = cast(np.stack([row[2][1] for row in rows]))
        ego = model.ego(ego_history, commands)["se2"]

    outputs = _split_sources(packed_output, source_lengths)
    forwards, evidences, contexts, local_actors = [], [], [], []
    source_offset = 0
    global_actor_parts = []

    for index, (row, output) in enumerate(zip(rows, outputs)):
        record, raw, ego_inputs, history = row
        poses = cast(raw["future_poses"]).double()
        prep, evidence, plan = prepare_projection(
            provider,
            record,
            raw,
            output,
            poses.detach().cpu().numpy(),
            history,
            use_gt_cache=True,
        )
        targets = source_targets(
            output, record, prep.state["current_pose"], prep.state["source_z_t0"]
        )
        yaws = effective_yaw(output, [int(c["class_id"]) for c in prep.state["current"]])
        mapped, context = continuous_context(
            evidence, plan, prep.state, targets, yaws, poses, provider.pcfg.grid
        )
        actor = cast(evidence.actor).long()
        global_actor_parts.append(
            torch.where(actor >= 0, actor + source_offset, actor)
        )
        source_offset += source_lengths[index]
        evidences.append(evidence)
        contexts.append(context)
        local_actors.append(actor)
        forwards.append(
            dict(
                output=output,
                ego=ego[index],
                ego_target=ego_inputs[2],
                poses=poses,
                targets=targets,
                yaws=yaws,
                prep=prep,
                evidence=evidence,
                plan=plan,
                mapped=mapped,
                field=field_views[index],
            )
        )

    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        encoded, _, point_lengths = _encode_repair_packed(
            model, field_views, reads, times, evidences
        )
        if sum(point_lengths):
            global_actor = torch.cat(global_actor_parts, 0)
            context = torch.cat(contexts, 0)
            base = cast(np.concatenate([fw["plan"].base for fw in forwards], 0))
            legal = cast(np.concatenate([fw["plan"].legal for fw in forwards], 0))
            actions = []
            for first in range(0, len(encoded), 8192):
                at = slice(first, first + 8192)
                actions.append(
                    model.columns(
                        encoded[at],
                        packed_output,
                        global_actor[at],
                        context[at],
                        base[at],
                        legal[at],
                    )
                )
            actions = torch.cat(actions)
        else:
            actions = fields.new_empty((0, 6, 2), dtype=torch.float32)

    start = 0
    for fw, length in zip(forwards, point_lengths):
        fw["actions"] = actions[start : start + length]
        start += length
    return forwards


def train_arm_packed(
    model,
    optimizer,
    provider,
    rows,
    mode,
    *,
    update,
    total_updates,
    object_weight=1.0,
    ego_weight=1.0,
    lovasz_weight=1.0,
    probe=False,
    optimized=True,
    horizon_group=2,
):
    """Pack B<=4 trainable network execution; preserve the generic REMOVE renderer."""
    if mode != "gt" or not optimized or not rows:
        raise ValueError("GT-only fused shared-history packed candidate required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    factor = 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * (update - 1) / max(1, total_updates)))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor

    begun = time.perf_counter()
    stages, terms, links = defaultdict(float), defaultdict(list), defaultdict(float)
    tick = time.perf_counter()
    forwards = forward_batch(model, provider, rows)
    stages["shared_encoder_motion_projection_ccr"] += time.perf_counter() - tick

    bridges, auxiliaries = [], []
    for row, fw in zip(rows, forwards):
        scene = WideGroupedScene(fw["prep"], provider.pcfg.grid, provider.device)
        bridge = RenderGradientBridge(fw)
        bridges.append(bridge)
        labels = np.asarray(row[1]["future_gt_occ"])

        for hs in _groups(horizon_group):
            tick = time.perf_counter()
            probability = scene.render_group(
                hs, **bridge.leaves, evidence=fw["evidence"], plan=fw["plan"]
            )
            losses, ce, lovasz = grouped_loss(
                probability, labels[hs], lovasz_weight=lovasz_weight
            )
            if not bool(torch.isfinite(losses).all()):
                raise RuntimeError("nonfinite full-grid occupancy objective")
            if probe and 1 in hs and losses.requires_grad:
                tensors, names = [fw["actions"]], ["occupancy_to_actions"]
                tensors += [fw["field"]]
                names += ["occupancy_to_shared_history"]
                if len(fw["output"]["residual_xy_m"]):
                    tensors += [
                        fw["output"]["residual_xy_m"],
                        fw["output"]["yaw_delta_rad"],
                    ]
                    names += ["occupancy_to_xy", "occupancy_to_yaw"]
                for name, grad in zip(
                    names, bridge.probe(losses[hs.index(1)], tensors)
                ):
                    links[name] += 0.0 if grad is None else float(grad.detach().float().norm())
            stages["soft_render_full_ce_lovasz"] += time.perf_counter() - tick

            tick = time.perf_counter()
            if losses.requires_grad:
                (losses / (len(rows) * 6)).sum().backward()
            stages["occupancy_backward"] += time.perf_counter() - tick
            terms["CE"].extend(ce.detach().unbind())
            terms["Lovasz"].extend(lovasz.detach().unbind())
            del probability, losses, ce, lovasz

        tick = time.perf_counter()
        geometry, existence = source_auxiliary(fw["output"], row[0], fw["prep"].state)
        ego_loss = ego_geometry_loss(
            fw["ego"][None], np.asarray(fw["ego_target"])[None]
        )
        auxiliary = object_weight * (geometry + existence) + ego_weight * ego_loss
        if not bool(torch.isfinite(auxiliary)):
            raise RuntimeError("nonfinite direct geometry objective")
        auxiliaries.append(auxiliary / len(rows))
        terms["source_geometry"].append(geometry.detach())
        terms["existence"].append(existence.detach())
        terms["ego_geometry"].append(ego_loss.detach())
        stages["shared_network_and_direct_geometry_backward"] += time.perf_counter() - tick

    # One multi-root traversal of the packed trainable graph.
    tick = time.perf_counter()
    roots, gradients = [torch.stack(auxiliaries).sum()], [None]
    for bridge in bridges:
        for key, leaf in bridge.leaves.items():
            if leaf.grad is not None:
                roots.append(bridge.original[key])
                gradients.append(leaf.grad)
    torch.autograd.backward(roots, grad_tensors=gradients)
    stages["shared_network_and_direct_geometry_backward"] += time.perf_counter() - tick

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
    tick = time.perf_counter()
    optimizer.step()
    if provider.device.type == "cuda":
        torch.cuda.synchronize(provider.device)
    stages["optimizer_finish"] += time.perf_counter() - tick

    result = {key: float(torch.stack(value).mean()) for key, value in terms.items()}
    result.update(
        mode=mode,
        windows=len(rows),
        sources=sum(len(fw["output"]["residual_xy_m"]) for fw in forwards),
        canonical_points=sum(len(fw["evidence"]) for fw in forwards),
        seconds=time.perf_counter() - begun,
        stages_seconds=dict(stages),
        grad_norm=float(norm),
        gradient_links=dict(links),
        training_backend=f"shared_history_packed_b{len(rows)}_h{horizon_group}",
        peak_memory_mib=(
            torch.cuda.max_memory_allocated(provider.device) / 2**20
            if provider.device.type == "cuda"
            else 0.0
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    return result
