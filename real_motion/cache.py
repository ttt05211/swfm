"""Versioned cache contract for Real-Motion OccFM."""
import torch
CACHE_VERSION="real_motion_v1"
REQUIRED_KEYS=("moving_history_latent","future_moving_latent","static_future_latent","kta_future_latent","generation_support")

def validate_sample(sample):
    missing=[k for k in REQUIRED_KEYS if k not in sample]
    if missing: raise KeyError(f"cache missing keys: {missing}")
    mh=sample["moving_history_latent"]; fm=sample["future_moving_latent"]
    st=sample["static_future_latent"]; kt=sample["kta_future_latent"]; ms=sample["generation_support"]
    for name,x in [("moving_history_latent",mh),("future_moving_latent",fm),("static_future_latent",st),("kta_future_latent",kt)]:
        if not torch.is_tensor(x) or x.ndim!=4: raise ValueError(f"{name} must be tensor [T,C,H,W]")
    if fm.shape!=st.shape or fm.shape!=kt.shape: raise ValueError("future moving/static/KTA latent shapes must match")
    if tuple(ms.shape)!=(fm.shape[0],fm.shape[-2],fm.shape[-1]): raise ValueError("generation_support must align with future latent")
    if mh.shape[1:]!=fm.shape[1:]: raise ValueError("history/future latent C,H,W must match")
    if "planning_support" in sample:
        ps=sample["planning_support"]
        if not torch.is_tensor(ps) or ps.ndim!=3: raise ValueError("planning_support must be [T,H,W]")
        if tuple(ps.shape[-2:])!=tuple(fm.shape[-2:]): raise ValueError("planning_support H,W must match latent map")
    return True

def save_cache(path,samples,metadata=None):
    for s in samples: validate_sample(s)
    torch.save({"version":CACHE_VERSION,"metadata":metadata or {},"samples":samples},path)

def load_cache(path):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("version")!=CACHE_VERSION: raise ValueError(f"unsupported cache version: {payload.get('version')}")
    for s in payload["samples"]: validate_sample(s)
    return payload
