"""Thin, explicit adapters around the pinned official OccFM implementation."""
from contextlib import contextmanager
from pathlib import Path
import os
import hashlib
import torch


def file_sha256(path, chunk_bytes=8*1024*1024):
    """Stable checkpoint fingerprint; computed only at setup/load time."""
    h=hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk=f.read(chunk_bytes)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def _pushd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_occfm_config(upstream_root, config_rel):
    """Load an official OccFM yaml while preserving its relative _BASE_CONFIG_."""
    upstream_root = Path(upstream_root).resolve()
    with _pushd(upstream_root):
        from easydict import EasyDict
        from forecast.config import cfg_from_yaml_file
        cfg = EasyDict()
        cfg_from_yaml_file(str(config_rel), cfg)
    return cfg


def load_official_vae(upstream_root, checkpoint, device="cuda",
                      config_rel="tools/cfgs/occfm_vae.yaml"):
    checkpoint = str(Path(checkpoint).resolve())
    cfg = load_occfm_config(upstream_root, config_rel)
    with _pushd(Path(upstream_root).resolve()):
        from forecast.models import build_network
        model = build_network(cfg.MODEL, cfg.LOSS)
    model.recover_training(checkpoint)
    model.eval().to(device)
    model.requires_grad_(False)
    return model, cfg


def load_official_wm(upstream_root, checkpoint, device="cuda",
                     config_rel="tools/cfgs/occfm.yaml"):
    checkpoint = str(Path(checkpoint).resolve())
    cfg = load_occfm_config(upstream_root, config_rel)
    with _pushd(Path(upstream_root).resolve()):
        from forecast.models import build_network
        model = build_network(cfg.MODEL, cfg.LOSS, cache_mode=True)
    model.recover_training(checkpoint)
    model.eval().to(device)
    model.requires_grad_(False)
    return model, cfg


class OccFMVAEAdapter:
    """Deterministic-control wrapper around the official stochastic VAE.

    Official ``VaeQuant`` samples ``mu + sigma * eps``. For fair decomposition
    comparisons we expose both ``mode='mean'`` and seeded ``mode='sample'``.
    The latter matches the training/cache distribution while making paired
    Full/Static/Moving P0 runs reproducible.
    """
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, semantic_occ, mode="sample", seed=None):
        x = torch.as_tensor(semantic_occ, device=self.device, dtype=torch.long)
        original_ndim = x.ndim
        if original_ndim == 3:  # H,W,D
            x = x.unsqueeze(0)
        if x.ndim not in (4, 5):
            raise ValueError("semantic_occ must be [H,W,D], [B,H,W,D] or [B,F,H,W,D]")
        video = x.ndim == 5
        B = x.shape[0]
        F = x.shape[1] if video else None

        batch = {"semantic_occ": x}
        old_e, old_enc = self.model.embedding.skip, self.model.encoder.skip
        try:
            self.model.embedding.skip = False
            self.model.encoder.skip = False
            batch = self.model.embedding(batch)
            batch = self.model.encoder(batch)
        finally:
            self.model.embedding.skip = old_e
            self.model.encoder.skip = old_enc

        compressed = batch["compressed_features"]
        dim = compressed.shape[1] // 2
        mu = compressed[:, :dim]
        sigma = torch.exp(compressed[:, dim:] / 2).to(compressed.dtype)
        if mode == "mean":
            latent = mu
        elif mode == "sample":
            gen = None
            if seed is not None:
                gen = torch.Generator(device=mu.device)
                gen.manual_seed(int(seed))
            eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=gen)
            latent = mu + sigma * eps
        else:
            raise ValueError("mode must be 'sample' or 'mean'")

        if video:
            latent = latent.reshape(B, F, *latent.shape[1:])
        elif original_ndim == 3:
            latent = latent[0]
        return latent

    @torch.no_grad()
    def decode_logits(self, latent):
        z = torch.as_tensor(latent, device=self.device)
        original_ndim = z.ndim
        if original_ndim == 3:
            z = z.unsqueeze(0)
        video = z.ndim == 5
        if z.ndim not in (4, 5):
            raise ValueError("latent must be [C,h,w], [B,C,h,w], or [B,F,C,h,w]")
        B = z.shape[0]
        F = z.shape[1] if video else None
        flat = z.reshape(-1, *z.shape[-3:]) if video else z
        batch = {"sampled_features": flat}
        old = self.model.decoder.skip
        try:
            self.model.decoder.skip = False
            batch = self.model.decoder(batch)
        finally:
            self.model.decoder.skip = old

        decoded = batch["decoded_map"]
        height = self.model.embedding.height_num
        cate = self.model.embedding.cate
        decoded = decoded.reshape(decoded.shape[0], height, cate, decoded.shape[-2], decoded.shape[-1])
        decoded = decoded.permute(0, 3, 4, 1, 2)  # b,h,w,d,c
        template = self.model.embedding.class_embeds.weight.T.unsqueeze(0).detach()
        logits = torch.matmul(decoded, template)  # b,h,w,d,num_classes
        if video:
            logits = logits.reshape(B, F, *logits.shape[1:])
        elif original_ndim == 3:
            logits = logits[0]
        return logits

    @torch.no_grad()
    def decode_labels(self, latent):
        return self.decode_logits(latent).argmax(dim=-1)

    @torch.no_grad()
    def empty_latent(self, shape_hwd=(200, 200, 16), free_label=17, mode="mean", seed=0):
        empty = torch.full(shape_hwd, int(free_label), device=self.device, dtype=torch.long)
        return self.encode(empty, mode=mode, seed=seed)


@torch.no_grad()
def run_frozen_occfm_forecast(wm, history_latent, future_reference_latent,
                              trajectory=None, seed=0, hist_last=4):
    """Run the pinned official OccFM sampler on one cached-latent window.

    ``future_reference_latent`` is used only to provide the official sampler's
    future tensor shape and loss target; generated prediction does not see its
    values because ``cfm_eval`` replaces future with fresh noise.
    """
    device = next(wm.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("official OccFM sampler contains .cuda(); run P0-A on CUDA")
    hist = torch.as_tensor(history_latent, device=device)
    fut = torch.as_tensor(future_reference_latent, device=device)
    if hist.ndim != 4 or fut.ndim != 4:
        raise ValueError("history/future latent must be [T,C,H,W]")
    hist = hist.clone()
    if hist_last is not None and hist_last < hist.shape[0]:
        hist[:hist.shape[0] - int(hist_last)] = 0
    x = torch.cat([hist, fut], dim=0).unsqueeze(0)
    traj = None if trajectory is None else torch.as_tensor(trajectory, device=device, dtype=x.dtype).unsqueeze(0)
    batch = {
        "x_sampled": x,
        "trajectory": traj,
        "paths": ["p0"],
        "cfm_eval": True,
    }
    cuda_devices = [device.index or 0]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        _, _, disp = wm(batch)
    pred = disp["pred_occ"]
    if pred.ndim >= 1 and pred.shape[0] == 1:
        pred = pred[0]
    return pred.detach().cpu()
