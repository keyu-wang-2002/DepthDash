"""Check every fused residual against its plain PyTorch path, on one CUDA GPU:

    python baselines/fused/check_fused.py [--layers 28 --dim 512]

For each baseline with a FUSED switch, it runs forward + backward of a small GPT through both paths from the
same weights and data, and compares the loss and every gradient (bf16, so expect agreement to ~1e-2, not
bit-exactness). It also times both paths.
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "baselines")]

from model import GPT  # noqa: E402
from train_gpt import init_backbone_  # noqa: E402

BASELINES = ["attnres", "block_attnres", "mhc"]


def load(name):
    spec = importlib.util.spec_from_file_location(f"check_{name}", ROOT / "baselines" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(module, args):
    torch.manual_seed(0)
    gpt = GPT(args.vocab, args.layers, args.dim, args.head_dim, module.Residual).cuda()
    init_backbone_(gpt)
    with torch.no_grad():
        for name, p in gpt.named_parameters():
            if name.endswith("proj.weight"):  # un-zero the output projections so every path carries gradient
                p.normal_(std=0.02)
        for p in gpt.residual.parameters():  # perturb the zero-init routing so it is exercised
            p.add_(torch.randn_like(p) * 0.05)
    return gpt


def run(gpt, x, y):
    gpt.zero_grad(set_to_none=True)
    loss = gpt(x, y) / y.numel()
    loss.backward()
    return loss.item(), {n: p.grad.float().clone() for n, p in gpt.named_parameters() if p.grad is not None}


def timed(gpt, x, y, iters=10):
    for _ in range(3):
        run(gpt, x, y)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run(gpt, x, y)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=50304)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=1024)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "the fused kernels need a CUDA GPU"

    torch.manual_seed(0)
    x = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, args.vocab, (args.batch, args.seq), device="cuda")
    ok = True
    for name in BASELINES:
        module = load(name)
        if not module.FUSED:
            print(f"SKIP {name}: its kernels are not importable here")
            continue
        gpt = build(module, args)
        loss_fused, grads_fused = run(gpt, x, y)
        ms_fused = timed(gpt, x, y)
        module.FUSED = False
        loss_plain, grads_plain = run(gpt, x, y)
        ms_plain = timed(gpt, x, y)
        assert grads_fused.keys() == grads_plain.keys(), "the two paths train different parameters"
        worst, worst_name = 0.0, None
        for n, g in grads_plain.items():
            err = ((grads_fused[n] - g).norm() / g.norm().clamp_min(1e-12)).item()
            if err > worst:
                worst, worst_name = err, n
        good = abs(loss_fused - loss_plain) < 1e-2 and worst < 5e-2
        ok &= good
        print(f"{'ok' if good else 'FAIL':4} {name:14} loss fused={loss_fused:.5f} plain={loss_plain:.5f}"
              f"  worst grad rel err={worst:.2e} ({worst_name})"
              f"  fwd+bwd fused={ms_fused:.1f}ms plain={ms_plain:.1f}ms")
    print("ALL OK" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
