from collections import OrderedDict
import torch

def _strip_module_prefix(state):
    return OrderedDict((k[7:] if k.startswith("module.") else k,v) for k,v in state.items())

def extract_state_dict(checkpoint):
    if isinstance(checkpoint,(str,bytes)):
        checkpoint=torch.load(checkpoint,map_location="cpu",weights_only=True)
    return _strip_module_prefix(checkpoint.get("state_dict",checkpoint))

def load_shape_safe(module: torch.nn.Module, checkpoint,
                    prefixes=("transition_model.",""), verbose=True):
    """Load only keys whose names and tensor shapes match."""
    src=extract_state_dict(checkpoint); dst=module.state_dict(); accepted={}; skipped={}
    for dk,dv in dst.items():
        candidates=[p+dk if p else dk for p in prefixes]
        found=next((k for k in candidates if k in src),None)
        if found is None: skipped[dk]="missing"; continue
        sv=src[found]
        if tuple(sv.shape)!=tuple(dv.shape):
            skipped[dk]=f"shape {tuple(sv.shape)} != {tuple(dv.shape)}"; continue
        accepted[dk]=sv
    incompatible=module.load_state_dict(accepted,strict=False)
    report={"loaded":len(accepted),"target_total":len(dst),"skipped":skipped,
            "missing_after_load":list(incompatible.missing_keys),
            "unexpected_after_load":list(incompatible.unexpected_keys)}
    if verbose:
        print(f"[shape-safe-load] loaded {len(accepted)}/{len(dst)} tensors; skipped {len(skipped)}")
        shape_skips=[(k,v) for k,v in skipped.items() if v.startswith("shape")]
        if shape_skips: print("[shape-safe-load] shape skips:",shape_skips[:10])
    return report
