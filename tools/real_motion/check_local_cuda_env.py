"""Small, real CUDA forward/backward check; not an FPS benchmark.

Run with the project's explicit .venv-cuda-check interpreter on Windows.
No checkpoint, dataset, global interpreter, or installed package is modified.
"""
import json
import platform
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    from real_motion.sparse_evidence_repair import SparseRepairHead

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: do not silently run this check on CPU")
    torch.set_num_threads(1)
    torch.manual_seed(20261005)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.reset_peak_memory_stats(device)
    report = dict(
        interpreter=sys.executable,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_runtime=torch.version.cuda,
        gpu=properties.name,
        compute_capability=list(torch.cuda.get_device_capability(device)),
        total_vram_mib=properties.total_memory / 2**20,
        bf16_supported=torch.cuda.is_bf16_supported(),
        checks={},
        scope="functional CUDA checks, NOT real-data quality or speed evidence",
    )
    # Real device allocation, cuBLAS operation, and autograd.
    x = torch.randn(64, 64, device=device, requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all():
        raise RuntimeError("CUDA matrix/backward produced nonfinite values")
    report["checks"]["cuda_matrix_backward"] = "PASS"
    del x, loss

    # The actual new repair head, six horizons, live source-context gradients,
    # and AdamW. Small batches fit the 4-GiB laptop GPU without taking all VRAM.
    for precision in ("float32", "bfloat16"):
        if precision == "bfloat16" and not report["bf16_supported"]:
            report["checks"][precision + "_sparse_head_backward"] = "UNSUPPORTED"
            continue
        model = SparseRepairHead("local_consensus").to(device)
        # This disposable instance is only a gradient probe. The production
        # conservative score initializer intentionally blocks context gradients
        # until its first update; nonzero diagnostic weights exercise the path.
        with torch.no_grad():
            model.score.weight.normal_(std=0.01)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        features = torch.randn(256, model.point_dim, device=device)
        actors = torch.arange(256, device=device) % 8
        actors[::9] = -2
        classes = torch.arange(256, device=device) % 17
        source = torch.randn(8, 128, device=device, requires_grad=True)
        future = torch.randn(8, 6, 128, device=device, requires_grad=True)
        target = (torch.rand(256, 6, device=device) < 0.1).float()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            source.grad = future.grad = None
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=precision == "bfloat16"):
                logits = model(features, actors, classes, source, future)
                if logits.shape != (256, 6):
                    raise RuntimeError("repair head must produce all six horizons")
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target)
            loss.backward()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite sparse training loss")
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise RuntimeError("missing/nonfinite sparse model gradients")
            if (source.grad is None or future.grad is None
                    or not torch.isfinite(source.grad).all()
                    or not torch.isfinite(future.grad).all()
                    or source.grad.abs().sum() == 0
                    or future.grad.abs().sum() == 0):
                raise RuntimeError("missing/nonfinite live source-context gradient")
            optimizer.step()
        torch.cuda.synchronize(device)
        report["checks"][precision + "_sparse_head_backward"] = "PASS"
        del model, optimizer, features, actors, classes, source, future, target, logits, loss, gradients
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
