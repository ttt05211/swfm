"""Read-only Shared-History execution benchmark.

Compares the CURRENT scientific trainer against full-six-supervision execution
candidates without publishing any training update:

  current_h2       original sequential network, H2 renderer
  sequential_h3    original sequential network, H3 renderer
  sequential_h6    original sequential network, H6 renderer
  packed_h2        packed global B<=4 trainable network, H2 renderer
  packed_h3        packed global B<=4 trainable network, H3 renderer
  packed_h6        packed global B<=4 trainable network, H6 renderer

REMOVE remains present in every candidate.  Losses, six future horizons, motion
and ego supervision are unchanged.  Every candidate is reset to the same source
weights/Adam/RNG before warmup and before each timed pass.
"""
import copy
import json
from pathlib import Path
import time

import torch

from tools.joint_occ_experiments import run as legacy
from tools.joint_occ_experiments.data import attach_caches, prefetched_batches
from tools.joint_occ_experiments.performance.probe import (
    legacy_fingerprint,
    state_parity,
)
from tools.shared_history_experiments import engine as reference
from tools.shared_history_experiments.history import SharedHistoryInputs
from tools.shared_history_experiments.model import SharedJoint, PROTOCOL
from .engine import train_arm_grouped, train_arm_packed


BACKENDS = (
    "current_h2",
    "sequential_h3",
    "sequential_h6",
    "packed_h2",
    "packed_h3",
    "packed_h6",
)


def source_fingerprint(root):
    """Recreate the exact shared pilot implementation fingerprint.

    The nested speed/ directory is intentionally excluded, so adding this
    read-only tool does not invalidate the running scientific checkpoint.
    """
    base = legacy_fingerprint(root)
    prototype = list((root / "tools/shared_history_experiments").glob("*.py")) + [
        root / "tools/joint_occ_experiments/batching/geometry.py",
        root / "tools/joint_occ_experiments/soft_inference/fast.py",
        root / "tools/joint_occ_experiments/soft_inference/engine.py",
        root / "real_motion/canonical_causal_repair.py",
        root / "real_motion/local_st_world_model.py",
    ]
    return legacy.stable_json_fingerprint(
        dict(
            base=base,
            variant="shared",
            prototype={
                str(path.relative_to(root)): legacy.sha256(path)
                for path in prototype
            },
        )
    )


def execution_fingerprint(root, backend):
    files = sorted(Path(__file__).parent.glob("*.py"))
    return legacy.stable_json_fingerprint(
        dict(
            source=source_fingerprint(root),
            backend=backend,
            protocol="shared_history_full6_execution_speed_v2",
            speed_sources={
                str(path.relative_to(root)): legacy.sha256(path) for path in files
            },
        )
    )


def _diagnostic(stats):
    keys = (
        "CE",
        "Lovasz",
        "source_geometry",
        "existence",
        "ego_geometry",
        "grad_norm",
    )
    return dict(
        losses={key: torch.tensor(stats[key]) for key in keys},
        gradient_links={
            key: torch.tensor(value)
            for key, value in stats["gradient_links"].items()
        },
    )


def measure(model, optimizer, provider, saved, batches, *, passes=2, progress=print):
    if not batches or passes < 2:
        raise ValueError("nonempty batches and at least two alternating passes required")
    weights = saved["models"]["gt"]
    adam = saved["optimizers"]["gt"]
    contract = saved["contract"]

    def reset():
        model.load_state_dict(weights, strict=True)
        optimizer.load_state_dict(copy.deepcopy(adam))
        legacy.restore_rng(saved["rng"])
        model.train().requires_grad_(True)

    def state():
        return dict(
            model={
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            adam=copy.deepcopy(optimizer.state_dict()),
        )

    def step(name, rows, index, probe=False):
        common = dict(
            update=min(saved["updates"] + index + 1, contract["total_updates"]),
            total_updates=contract["total_updates"],
            object_weight=contract["weights"][0],
            ego_weight=contract["weights"][1],
            lovasz_weight=contract["weights"][2],
            probe=probe,
            optimized=True,
        )
        if name == "current_h2":
            return reference.train_arm(
                model, optimizer, provider, rows, "gt", **common
            )
        if name.startswith("sequential_h"):
            return train_arm_grouped(
                model,
                optimizer,
                provider,
                rows,
                "gt",
                horizon_group=int(name.rsplit("h", 1)[1]),
                **common,
            )
        if name.startswith("packed_h"):
            return train_arm_packed(
                model,
                optimizer,
                provider,
                rows,
                "gt",
                horizon_group=int(name.rsplit("h", 1)[1]),
                **common,
            )
        raise ValueError("unknown backend " + name)

    warm = {}
    warm_reference = None
    reference_stats = None
    for name in BACKENDS:
        reset()
        stats = step(name, batches[0], 0, probe=True)
        current = state()
        if warm_reference is None:
            warm_reference = current
            reference_stats = stats
        state_check = state_parity(warm_reference, current)
        diagnostic_check = state_parity(
            _diagnostic(reference_stats), _diagnostic(stats)
        )
        warm[name] = dict(
            numerical_parity=(
                state_check["numerical_parity"]
                and diagnostic_check["numerical_parity"]
            ),
            state_parity=state_check,
            loss_and_links_parity=diagnostic_check,
            gradient_links=stats["gradient_links"],
            training_backend=stats["training_backend"],
        )
        progress(
            "SHARED_SPEED_WARM "
            + json.dumps(dict(backend=name, **warm[name])),
            flush=True,
        )

    timed = {name: [] for name in BACKENDS}
    parity = {
        name: dict(numerical_parity=True, max_state_abs_difference=0.0)
        for name in BACKENDS
    }

    for repeat in range(passes):
        order = BACKENDS if repeat % 2 == 0 else BACKENDS[::-1]
        pass_reference = None
        for name in order:
            reset()
            if provider.device.type == "cuda":
                torch.cuda.synchronize(provider.device)
                torch.cuda.reset_peak_memory_stats(provider.device)
            tick = time.perf_counter()
            reports = [
                step(name, rows, index)
                for index, rows in enumerate(batches)
            ]
            if provider.device.type == "cuda":
                torch.cuda.synchronize(provider.device)
            elapsed = time.perf_counter() - tick
            windows = sum(report["windows"] for report in reports)
            current = state()
            if pass_reference is None:
                # The first candidate in each alternating pass is only a
                # temporary reference; all candidates started from identical
                # source state/RNG and consumed identical batches.
                pass_reference = current

            # Compare final state to current_h2 explicitly below.  Store timing now.
            row = dict(
                pass_index=repeat + 1,
                windows=windows,
                seconds=elapsed,
                seconds_per_window=elapsed / windows,
                peak_allocated_mib=(
                    torch.cuda.max_memory_allocated(provider.device) / 2**20
                    if provider.device.type == "cuda"
                    else None
                ),
                peak_reserved_mib=(
                    torch.cuda.max_memory_reserved(provider.device) / 2**20
                    if provider.device.type == "cuda"
                    else None
                ),
                host_stages_seconds_per_window={
                    stage: sum(
                        report["stages_seconds"].get(stage, 0.0)
                        for report in reports
                    )
                    / windows
                    for stage in reports[0]["stages_seconds"]
                },
                final_state=current,
            )
            timed[name].append(row)
            progress(
                "SHARED_SPEED "
                + json.dumps(
                    {
                        key: value
                        for key, value in dict(backend=name, **row).items()
                        if key != "final_state"
                    }
                ),
                flush=True,
            )

        # Compare all candidates to current_h2 from THIS repeat, independent of
        # AB/BA timing order.
        reference_state = timed["current_h2"][-1]["final_state"]
        for name in BACKENDS:
            check = state_parity(reference_state, timed[name][-1]["final_state"])
            parity[name]["numerical_parity"] &= check["numerical_parity"]
            parity[name]["max_state_abs_difference"] = max(
                parity[name]["max_state_abs_difference"],
                check["max_state_abs_difference"],
            )
            timed[name][-1]["final_state_parity"] = check

    # Do not serialize model/Adam snapshots into speed.json.
    for rows in timed.values():
        for row in rows:
            row.pop("final_state", None)

    means = {
        name: sum(row["seconds"] for row in rows)
        / sum(row["windows"] for row in rows)
        for name, rows in timed.items()
    }
    capacity = (
        torch.cuda.get_device_properties(provider.device).total_memory / 2**20
        if provider.device.type == "cuda"
        else None
    )
    candidates = {}
    baseline = "current_h2"
    for name in BACKENDS[1:]:
        faster_each_pass = all(
            candidate["seconds"] < reference_row["seconds"]
            for candidate, reference_row in zip(timed[name], timed[baseline])
        )
        numerical = (
            warm[name]["numerical_parity"]
            and parity[name]["numerical_parity"]
        )
        replay = parity[baseline]["numerical_parity"]
        peak_reserved = max(
            (
                row["peak_reserved_mib"]
                for row in timed[name]
                if row["peak_reserved_mib"] is not None
            ),
            default=None,
        )
        headroom = (
            capacity is not None
            and peak_reserved is not None
            and peak_reserved <= 0.9 * capacity
        )
        speedup = means[baseline] / means[name]
        candidates[name] = dict(
            speedup=speedup,
            numerical_parity=numerical,
            baseline_repeat_parity=replay,
            faster_each_pass=faster_each_pass,
            memory_headroom_ge_10percent=headroom,
            execution_fingerprint=execution_fingerprint(
                Path(__file__).resolve().parents[3], name
            ),
            # No arbitrary speed multiplier is baked into correctness.  The
            # fastest eligible candidate is selected only after real timing.
            pass_gate=bool(numerical and replay and faster_each_pass and headroom),
        )

    eligible = [
        name for name, value in candidates.items() if value["pass_gate"]
    ]
    recommended = min(eligible, key=lambda name: means[name]) if eligible else None
    reset()
    return dict(
        seconds_per_window=means,
        timed_passes=timed,
        warmup=warm,
        final_state_parity=parity,
        candidates=candidates,
        recommended_backend=recommended,
        scope=(
            "GT Shared-History real forward/backward+Adam; full six future "
            "CE+Lovasz, motion and ego supervision; REMOVE preserved; input I/O excluded"
        ),
        source_weights_optimizer_rng_unchanged=True,
        automatic_promotion=False,
    )


def main(argv=None):
    parser = legacy.parser()
    parser.description = __doc__
    parser.add_argument("--probe-steps", type=int, default=8)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args(argv)

    if (
        not args.resume
        or not 1 <= args.probe_steps <= 32
        or not 2 <= args.passes <= 4
    ):
        raise ValueError("source checkpoint, probe-steps 1..32 and passes 2..4 required")
    if args.mode != "gt" or args.training_backend != "fused" or args.evaluate_only:
        raise ValueError("GT fused training probe only")
    if args.window_batch_size != 4 or args.source_budget != 128:
        raise ValueError("probe keeps original global batch4/source128")

    out = Path(args.out_dir).resolve()
    source_checkpoint = Path(args.resume).resolve()
    if out == source_checkpoint.parent or source_checkpoint.parent in out.parents:
        raise ValueError("probe output must be OUTSIDE source training directory")
    if out.exists():
        raise RuntimeError("NEW probe directory required")

    device = legacy.require_cuda(args.device)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    out.mkdir(parents=True)

    from tools.real_motion.joint_training_recovery import snapshot_checkpoint

    digest = snapshot_checkpoint(source_checkpoint, out / "source_snapshot.pt")
    saved = torch.load(out / "source_snapshot.pt", map_location="cpu", weights_only=False)
    contract = saved["contract"]
    expected_protocol = PROTOCOL + "_shared"
    if (
        saved.get("protocol") != expected_protocol
        or contract.get("modes") != ["gt"]
        or contract.get("training_backend") != "fused"
        or contract.get("implementation") != source_fingerprint(root)
    ):
        raise RuntimeError("exact current Shared-History shared checkpoint/code required")
    if (
        contract["execution"]["torch_version"] != str(torch.__version__)
        or contract["execution"]["cuda_version"] != torch.version.cuda
    ):
        raise RuntimeError("original Torch/CUDA precision environment required")
    for key, expected in contract["data"].items():
        if legacy.sha256(getattr(args, key)) != expected:
            raise RuntimeError("source data differs: " + key)

    config = legacy.config_from_mapping_v17(contract["model"]["motion"])
    shared = contract["model"]["shared_history"]
    model = SharedJoint(
        config,
        width=int(shared["width"]),
        z_bins=int(shared["z_bins"]),
    ).to(device)
    if model.configs() != contract["model"]:
        raise RuntimeError("model architecture differs from checkpoint contract")

    optimizer = torch.optim.AdamW(
        [
            dict(
                params=model.transport.parameters(),
                lr=contract["lr"][0],
                initial_lr=contract["lr"][0],
                weight_decay=1e-4,
            ),
            dict(
                params=model.columns.parameters(),
                lr=contract["lr"][1],
                initial_lr=contract["lr"][1],
                weight_decay=0.01,
            ),
            dict(
                params=model.ego.parameters(),
                lr=contract["lr"][2],
                initial_lr=contract["lr"][2],
                weight_decay=0.01,
            ),
        ]
    )

    pcfg = legacy.make_prepare_config(
        legacy.load_runtime_config(args.config, args.override)
    )
    provider = legacy.PilotProvider(
        args.base_checkpoint,
        legacy.CLEAN_SHA256,
        pcfg,
        device,
        args.cpu_workers,
        model,
        None,
    )
    provider.raw_io_workers = 1

    from real_motion.canonical_repair_execution import CanonicalCpuExecution
    from real_motion.surface_projection_execution import SurfaceMapExecution
    from real_motion.strong_majority_execution import ParallelNativeMajority

    provider.ccr_execution = SurfaceMapExecution(
        CanonicalCpuExecution("native_parallel", min(4, args.cpu_workers))
    )
    provider.strong_majority = ParallelNativeMajority(min(4, args.cpu_workers))
    train_source = legacy.CachedColumnSource(
        legacy.NuScenesWindowSource(args.dataroot, info_pkl=args.train_info),
        args.frame_cache_mib,
        copy_on_insert=False,
    )
    provider.ccr_history_cache_source = train_source
    provider.ccr_val_history_cache_source = None

    try:
        caches = attach_caches(provider, args, root)
        if {
            key: caches[key]["namespace"] for key in ("train", "val")
        } != contract["cache_namespace"]:
            raise RuntimeError("history cache namespaces differ")

        _, records = legacy.load_cache(args.train_cache)
        legacy.record_keys(records)
        train = legacy.align_records(
            records, [tuple(key) for key in contract["train_keys"]]
        )
        del records

        groups = [
            legacy.epoch_groups(
                train,
                contract["seed"],
                epoch,
                contract["window_batch"],
                contract["source_budget"],
            )
            for epoch in range(contract["epochs"])
        ]
        if sum(len(group) for group in groups) != contract["total_updates"]:
            raise RuntimeError("source schedule/groups differ")

        epoch, cursor = int(saved["epoch"]), int(saved["batch"])
        if (
            not 0 <= epoch <= len(groups)
            or epoch == len(groups)
            and cursor != 0
            or epoch < len(groups)
            and not 0 <= cursor < len(groups[epoch])
            or saved["updates"]
            != sum(len(group) for group in groups[:epoch]) + cursor
        ):
            raise RuntimeError("invalid source cursor")

        future = [
            group
            for e, sequence in enumerate(groups)
            if e >= epoch
            for j, group in enumerate(sequence)
            if e > epoch or j >= cursor
        ]
        # A completed source run still has a deterministic probe population.
        jobs = (future + groups[0])[: args.probe_steps]
        descriptors = SharedHistoryInputs(
            provider, ram_mib=args.descriptor_ram_mib
        )
        batches = list(
            prefetched_batches(
                provider,
                train_source,
                jobs,
                descriptors,
                workers=args.prefetch_workers,
            )
        )

        def progress(message, *, flush=True):
            event, _, payload = message.partition(" ")
            with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(dict(event=event, **json.loads(payload))) + "\n"
                )
            print(message, flush=flush)

        report = measure(
            model,
            optimizer,
            provider,
            saved,
            batches,
            passes=args.passes,
            progress=progress,
        )
        report.update(
            status="complete",
            source_checkpoint=str(source_checkpoint),
            source_snapshot_sha256=digest,
            source_completed_updates=saved["updates"],
            scientific_updates_published=0,
            cache_audit=caches,
            probe_keys=[
                [record["scene_name"], record["t0_token"]]
                for group in jobs
                for record in group
            ],
            implementation=legacy.stable_json_fingerprint(
                {
                    path.name: legacy.sha256(path)
                    for path in Path(__file__).parent.glob("*.py")
                }
            ),
        )
        legacy.write_json(out / "speed.json", report)

        lines = [
            "===== SHARED-HISTORY FULL6 EXECUTION SPEED =====",
            f"source_update={saved['updates']}; scientific_updates_published=0",
            "REMOVE preserved; full six-horizon CE+Lovasz + motion + ego preserved.",
            "Same source weights/Adam/RNG/batches; warmup excluded; input I/O excluded.",
        ]
        for name in BACKENDS:
            lines.append(
                f"{name}: {report['seconds_per_window'][name]:.6f} seconds/window"
            )
        lines.append("recommended_backend=" + str(report["recommended_backend"]))
        for name, value in report["candidates"].items():
            lines.append("CANDIDATE " + name + " " + json.dumps(value, sort_keys=True))
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        return report
    finally:
        for name in ("ccr_history_cache", "ccr_val_history_cache"):
            cache = getattr(provider, name, None)
            if cache is not None and hasattr(cache, "close"):
                cache.close()


if __name__ == "__main__":
    main()
