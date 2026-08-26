#!/usr/bin/env python3
"""Sample sparse future moving latents with one global noise canvas per batch."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "upstream_occfm"
sys.path[:0] = [str(UP), str(ROOT)]

import torch
from torch.utils.data import DataLoader
from real_motion.dataset import RealMotionCacheDataset, collate_real_motion
from real_motion.windows import WindowPlanner, crop_windows, scatter_windows, window_coverage
from real_motion.models import MotionWindowFlowMatching, RealMotionWindowCFM


def make_model(window):
    tr = MotionWindowFlowMatching(
        in_channels=16, out_channels=16, model_channels=128, channel_multi=[2, 4],
        input_size=[window, window], trajectory_length=6, init_kernel_size=7,
        init_3d_conv_channels=64, attn_dim=32, temporal_attn_head=8,
        spatial_attn_head=8, prior_channels=32,
    )
    return RealMotionWindowCFM(tr)


def load_empty(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("empty_latent", obj.get("latent"))
    if not torch.is_tensor(obj) or obj.ndim != 3:
        raise ValueError("empty latent must be [C,H,W]")
    return obj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--empty-latent", default=None, help="optional override; otherwise use tensor stored in checkpoint")
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-window-coverage", type=float, default=0.95)
    p.add_argument("--allow-low-coverage", action="store_true")
    a = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = RealMotionCacheDataset(a.cache)
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                    collate_fn=collate_real_motion, drop_last=False)
    model = make_model(a.window).to(dev)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    if a.empty_latent is not None:
        empty = load_empty(a.empty_latent)
    else:
        empty = ck.get("empty_latent")
        if empty is None:
            raise KeyError("checkpoint lacks empty_latent; pass --empty-latent explicitly")
    planner = WindowPlanner((a.window, a.window), a.max_windows)

    outputs = []
    generator = torch.Generator(device=dev)
    generator.manual_seed(a.seed)
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            plan = planner.plan(batch["generation_support"],
                                context_support=batch.get("planning_support"))
            cov = window_coverage(batch["generation_support"], plan)
            min_cov = float(cov.min())
            if min_cov < a.min_window_coverage:
                msg = (f"future generation-support coverage {min_cov:.3f} < "
                       f"{a.min_window_coverage:.3f}; sampling would miss target regions")
                if not a.allow_low_coverage:
                    raise RuntimeError(msg)
                print("[WARN]", msg)

            hist = crop_windows(batch["moving_history_latent"], plan)
            sta = crop_windows(batch["static_future_latent"], plan)
            kta = crop_windows(batch["kta_future_latent"], plan)
            active = crop_windows(batch["generation_support"].unsqueeze(2), plan)
            B, K = hist.shape[:2]
            F = sta.shape[2]
            C = sta.shape[3]
            H, W = plan.full_hw
            valid = plan.valid.reshape(-1)

            empty_full = empty.to(dev, dtype=sta.dtype)[None, None].expand(B, F, -1, -1, -1)
            empty_win = crop_windows(empty_full, plan)
            # Critical: one global stochastic canvas. Overlapping windows crop the
            # same initial z0 for each shared global cell.
            global_noise = torch.randn((B, F, C, H, W), device=dev, dtype=sta.dtype,
                                       generator=generator)
            noise_win = crop_windows(global_noise, plan)

            if not bool(valid.any()):
                outputs.append(empty_full.cpu())
                continue

            def flat(x):
                return x.reshape(B*K, *x.shape[2:])[valid]

            fhist, fsta, fkta, factive, fempty, fnoise = map(
                flat, (hist, sta, kta, active, empty_win, noise_win)
            )
            origins = plan.origins.reshape(B*K, 2)[valid]
            prior = torch.cat([fsta, fkta], dim=2)
            traj = batch.get("trajectory")
            if traj is not None:
                traj = traj[:, None].expand(B, K, *traj.shape[1:]).reshape(
                    B*K, *traj.shape[1:]
                )[valid]

            pred = model.sample(
                fhist, tuple(fsta.shape[:2]) + (C, a.window, a.window), prior,
                factive, known_future=fempty, trajectory=traj,
                window_origins=origins, initial_noise=fnoise,
            )
            # The sampler clamps outside support at every ODE step. Assert this
            # invariant again before scatter so no unsupervised margin can leak.
            mask_c = factive.bool().expand(-1, -1, C, -1, -1)
            pred = torch.where(mask_c, pred, fempty)

            pred_pad = torch.zeros(B*K, F, C, a.window, a.window,
                                   device=dev, dtype=pred.dtype)
            pred_pad[valid] = pred
            pred_pad = pred_pad.reshape(B, K, F, C, a.window, a.window)
            full = scatter_windows(pred_pad, plan, base=empty_full.to(pred.dtype))
            outputs.append(full.cpu())

    out = torch.cat(outputs, dim=0) if outputs else torch.empty(0)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"future_dynamic_latent": out, "seed": a.seed}, a.output)
    print("saved", a.output, tuple(out.shape))


if __name__ == "__main__":
    main()
