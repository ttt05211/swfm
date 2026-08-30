"""Thin, explicit adapters around the pinned official OccFM implementation."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
from collections.abc import Sequence

import torch


def file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
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
    upstream_root = Path(upstream_root).resolve()
    with _pushd(upstream_root):
        from easydict import EasyDict
        from forecast.config import cfg_from_yaml_file

        cfg = EasyDict()
        cfg_from_yaml_file(str(config_rel), cfg)
    return cfg


def load_official_vae(
    upstream_root,
    checkpoint,
    device="cuda",
    config_rel="tools/cfgs/occfm_vae.yaml",
):
    checkpoint = str(Path(checkpoint).resolve())
    cfg = load_occfm_config(upstream_root, config_rel)
    with _pushd(Path(upstream_root).resolve()):
        from forecast.models import build_network

        model = build_network(cfg.MODEL, cfg.LOSS)
    model.recover_training(checkpoint)
    model.eval().to(device)
    model.requires_grad_(False)
    return model, cfg


def load_official_wm(
    upstream_root,
    checkpoint,
    device="cuda",
    config_rel="tools/cfgs/occfm_fut.yaml",
):
    """Load the official future-trajectory OccFM release (epoch=000196 family)."""
    checkpoint = str(Path(checkpoint).resolve())
    cfg = load_occfm_config(upstream_root, config_rel)
    with _pushd(Path(upstream_root).resolve()):
        from forecast.models import build_network

        model = build_network(cfg.MODEL, cfg.LOSS, cache_mode=True)
    model.recover_training(checkpoint)
    model.eval().to(device)
    model.requires_grad_(False)
    return model, cfg


def _seeded_eps_like(mu, seed, logical_batch, frames_per_sample):
    if seed is None:
        return torch.randn_like(mu)
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        seeds = [int(s) for s in seed]
        if len(seeds) != int(logical_batch):
            raise ValueError(
                f"per-sample seed length {len(seeds)} != logical batch {logical_batch}"
            )
        expected = int(logical_batch) * int(frames_per_sample)
        if mu.shape[0] != expected:
            raise ValueError(
                f"latent batch {mu.shape[0]} incompatible with "
                f"{logical_batch}x{frames_per_sample}"
            )
        chunks = []
        for i, s in enumerate(seeds):
            gen = torch.Generator(device=mu.device)
            gen.manual_seed(s)
            part = mu[i * frames_per_sample : (i + 1) * frames_per_sample]
            chunks.append(
                torch.randn(
                    part.shape,
                    device=mu.device,
                    dtype=mu.dtype,
                    generator=gen,
                )
            )
        return torch.cat(chunks, dim=0)
    gen = torch.Generator(device=mu.device)
    gen.manual_seed(int(seed))
    return torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=gen)


class OccFMVAEAdapter:
    def __init__(self, model):
        self.model = model

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def encode(self, semantic_occ, mode="sample", seed=None):
        x = torch.as_tensor(semantic_occ)
        if x.device != self.device or x.dtype != torch.long:
            x = x.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=(x.device.type == "cpu" and x.is_pinned()),
            )
        original_ndim = x.ndim
        if original_ndim == 3:
            x = x.unsqueeze(0)
        if x.ndim not in (4, 5):
            raise ValueError(
                "semantic_occ must be [H,W,D], [B,H,W,D] or [B,F,H,W,D]"
            )
        video = x.ndim == 5
        B = x.shape[0]
        F = x.shape[1] if video else 1
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
            latent = mu + sigma * _seeded_eps_like(mu, seed, B, F)
        else:
            raise ValueError("mode must be 'sample' or 'mean'")
        if video:
            latent = latent.reshape(B, F, *latent.shape[1:])
        elif original_ndim == 3:
            latent = latent[0]
        return latent

    def _decode_features(self, latent):
        """Decode latent to per-voxel embedding features while preserving autograd.

        The official VAE parameters are frozen by ``load_official_vae``.  This
        helper deliberately does *not* use ``torch.no_grad`` so gradients can
        flow from a semantic loss through the frozen decoder back to the input
        latent produced by the World Model.
        """
        if not torch.is_tensor(latent):
            z = torch.as_tensor(latent, device=self.device)
        else:
            z = latent.to(device=self.device)
        original_ndim = z.ndim
        if original_ndim == 3:
            z = z.unsqueeze(0)
        video = z.ndim == 5
        if z.ndim not in (4, 5):
            raise ValueError(
                "latent must be [C,h,w], [B,C,h,w], or [B,F,C,h,w]"
            )
        B = z.shape[0]
        F = z.shape[1] if video else 1
        flat = z.reshape(-1, *z.shape[-3:]) if video else z
        batch = {"sampled_features": flat}
        old = self.model.decoder.skip
        try:
            self.model.decoder.skip = False
            batch = self.model.decoder(batch)
        finally:
            self.model.decoder.skip = old
        decoded = batch["decoded_map"]
        height = int(self.model.embedding.height_num)
        cate = int(self.model.embedding.cate)
        decoded = decoded.reshape(
            decoded.shape[0], height, cate, decoded.shape[-2], decoded.shape[-1]
        )
        decoded = decoded.permute(0, 3, 4, 1, 2)
        if video:
            decoded = decoded.reshape(B, F, *decoded.shape[1:])
        elif original_ndim == 3:
            decoded = decoded[0]
        return decoded

    def _class_template(self):
        return self.model.embedding.class_embeds.weight.T.detach()

    def decode_logits_with_grad(self, latent):
        """Full semantic logits with gradient only to the input latent."""
        decoded = self._decode_features(latent)
        return torch.matmul(decoded, self._class_template())

    def decode_logits_at_flat_indices(self, latent, flat_indices_per_sample):
        """Project semantic logits only at sparse 4D voxel indices.

        ``latent`` must be [B,F,C,h,w]. Each item in ``flat_indices_per_sample``
        indexes the flattened [F,X,Y,Z] decoded voxel grid for that sample.  The
        expensive frozen decoder still runs on the complete latent so its exact
        receptive-field contract is preserved, but the class-embedding matmul is
        performed only at supervised voxels. Returned tensors keep autograd to
        ``latent`` and have shape [N_i,num_semantic_classes].
        """
        if not torch.is_tensor(latent) or latent.ndim != 5:
            raise ValueError("sparse decoder supervision requires latent [B,F,C,h,w]")
        if len(flat_indices_per_sample) != int(latent.shape[0]):
            raise ValueError("one sparse-index tensor is required per latent sample")
        decoded = self._decode_features(latent)
        if decoded.ndim != 6:
            raise RuntimeError("video decoder features must be [B,F,X,Y,Z,D]")
        B = int(decoded.shape[0])
        flat_size = 1
        for dim in decoded.shape[1:-1]:
            flat_size *= int(dim)
        template = self._class_template()
        out = []
        for b in range(B):
            idx = torch.as_tensor(
                flat_indices_per_sample[b], device=decoded.device, dtype=torch.long
            ).reshape(-1)
            if idx.numel() == 0:
                out.append(decoded.new_zeros((0, template.shape[-1])))
                continue
            if int(idx.min()) < 0 or int(idx.max()) >= flat_size:
                raise ValueError(
                    f"sparse decoder index out of range [0,{flat_size}): "
                    f"min={int(idx.min())} max={int(idx.max())}"
                )
            feat = decoded[b].reshape(flat_size, decoded.shape[-1]).index_select(0, idx)
            out.append(torch.matmul(feat, template))
        return out

    @torch.no_grad()
    def decode_logits(self, latent):
        return self.decode_logits_with_grad(latent)

    @torch.no_grad()
    def decode_labels(self, latent):
        return self.decode_logits(latent).argmax(dim=-1)

    @torch.no_grad()
    def empty_latent(self, shape_hwd=(200, 200, 16), free_label=17, mode="mean", seed=0):
        empty = torch.full(
            shape_hwd,
            int(free_label),
            device=self.device,
            dtype=torch.long,
        )
        return self.encode(empty, mode=mode, seed=seed)


@torch.no_grad()
def run_frozen_occfm_forecast(
    wm,
    history_latent,
    future_reference_latent,
    trajectory=None,
    seed=0,
    hist_last=4,
):
    """Run the official OccFM-fut 196 protocol for the P0 dense diagnostic."""
    device = next(wm.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("official OccFM sampler contains .cuda(); run P0-A on CUDA")
    if int(hist_last) != 4:
        raise ValueError("official OccFM-fut release uses HIST_LAST=4")
    hist = torch.as_tensor(history_latent, device=device)
    fut = torch.as_tensor(future_reference_latent, device=device)
    if hist.ndim != 4 or fut.ndim != 4:
        raise ValueError("history/future latent must be [T,C,H,W]")
    hist = hist.clone()
    if hist_last < hist.shape[0]:
        hist[: hist.shape[0] - int(hist_last)] = 0
    x = torch.cat([hist, fut], dim=0).unsqueeze(0)
    if trajectory is None:
        raise ValueError("OccFM-fut 196 requires the official [12,2] GT ego trajectory")
    traj = torch.as_tensor(trajectory, device=device, dtype=x.dtype)
    if traj.shape != (12, 2):
        raise ValueError(f"OccFM-fut trajectory must be [12,2], got {tuple(traj.shape)}")
    traj = traj.unsqueeze(0)
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
