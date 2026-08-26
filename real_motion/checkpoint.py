from collections import OrderedDict
import os
import torch


def _strip_module_prefix(state):
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k,v in state.items())


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, (str, bytes, os.PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    return _strip_module_prefix(state)


def load_shape_safe(module: torch.nn.Module, checkpoint,
                    prefixes=("transition_model.", ""),
                    verbose=True):
    """Load only keys whose names *and tensor shapes* match."""
    src = extract_state_dict(checkpoint)
    dst = module.state_dict()
    accepted, skipped = {}, {}
    for dk, dv in dst.items():
        candidates = [(p + dk if p else dk) for p in prefixes]
        found = next((k for k in candidates if k in src), None)
        if found is None:
            skipped[dk] = "missing"
            continue
        sv = src[found]
        if tuple(sv.shape) != tuple(dv.shape):
            skipped[dk] = f"shape {tuple(sv.shape)} != {tuple(dv.shape)}"
            continue
        accepted[dk] = sv
    incompatible = module.load_state_dict(accepted, strict=False)
    report = {
        "loaded": len(accepted),
        "target_total": len(dst),
        "loaded_keys": sorted(accepted),
        "skipped": skipped,
        "missing_after_load": list(incompatible.missing_keys),
        "unexpected_after_load": list(incompatible.unexpected_keys),
    }
    if verbose:
        print(f"[shape-safe-load] loaded {len(accepted)}/{len(dst)} tensors; skipped {len(skipped)}")
        shape_skips=[(k,v) for k,v in skipped.items() if v.startswith("shape")]
        if shape_skips:
            print("[shape-safe-load] shape skips:", shape_skips[:10])
    return report


def require_checkpoint_reuse(report, min_fraction=0.95,
                             required_prefixes=(
                                 "init_conv.", "init_temporal_attn.", "t_embedder.",
                                 "traj_encoder.", "downs.", "mid_", "ups.", "final_conv.",
                             )):
    """Turn shape-safe loading into a formal pre-training gate.

    ``pos_embed`` is expected to change for 20x20 windows and ``prior_proj`` is
    new/zero-initialized. Everything else should reuse the official transition
    checkpoint. A bad prefix or wrong checkpoint must stop training rather than
    merely printing a low load ratio.
    """
    total=max(int(report.get("target_total",0)),1)
    loaded=int(report.get("loaded",0))
    fraction=loaded/total
    if fraction < float(min_fraction):
        raise RuntimeError(
            f"upstream transition reuse {loaded}/{total}={fraction:.3f} < required {float(min_fraction):.3f}"
        )
    keys=set(report.get("loaded_keys",()))
    missing=[p for p in required_prefixes if not any(k.startswith(p) for k in keys)]
    if missing:
        raise RuntimeError(f"upstream checkpoint did not initialize critical transition blocks: {missing}")
    return fraction
