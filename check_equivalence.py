"""Check that a fast model (--model) computes the same function as its reference. Frozen.

    python check_equivalence.py --track deep --residual <dir>/residual.py --model <dir>/fast_model.py

The reference is the frozen model.py GPT with the submission's Residual: black-box sublayers, each called once,
in order. The fast model is the submission's `class GPT(model.GPT)`, which may compute the same function any way
it likes. This script, on one GPU:
  1. requires the same parameter names and shapes as the reference, so init and optimizer groups are unchanged;
  2. runs both in fp32 (TF32 off) from the same weights on the same batch, and compares every gradient (which
     depend on the whole forward pass too): at the init, and at --draws random weight draws at a trained scale,
     where every path through the function carries signal.

In fp32, a reordered or fused but correct implementation moves the gradients by ~1e-6. The tolerances sit far
above that, so a correct fast model passes with a wide margin, while a different function (a wrong scale, a
missing transpose, a detached input) fails. The check is there to catch honest mistakes; the rules are a
gentlemen's agreement. A fast model must therefore also run in fp32: write kernels for the input dtype rather
than hard-coding bf16.
"""
import argparse
import math
import sys
from pathlib import Path

import torch

from model import GPT
from tracks import SEQ_LEN, TRACKS, TRAIN_FILES, VOCAB_SIZE
from train_gpt import _load_data_shard, check_same_parameters, init_backbone_, load_model, load_residual

# measured on correct fast models: ~1e-6 overall and <= ~3e-4 for the worst tensor
GRAD_RTOL = 1e-4     # ||g_fast - g_ref|| / ||g_ref|| over all parameters
TENSOR_RTOL = 1e-2   # the same for each parameter tensor (tensors with a negligible gradient count only above)


def gradients(model, inputs, targets):
    model.zero_grad(set_to_none=True)
    model(inputs, targets).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grads


@torch.no_grad()
def randomize_(model: GPT, seed: int):
    """Every weight at a typical trained scale, unlike the init (zero output projections, identity routing)."""
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        if name.startswith("residual."):
            p.add_(0.1 * torch.randn_like(p))
        elif name.endswith("gains"):
            p.copy_(1 + 0.1 * torch.randn_like(p))
        elif p.ndim >= 2 and name != "embed.weight":
            p.copy_(torch.randn_like(p) * p.size(-1) ** -0.5)
        elif p.ndim < 2:
            p.copy_(0.02 * torch.randn_like(p))


def compare(reference, fast, inputs, targets):
    """Gradient errors of the fast model relative to the reference, with the reference's weights loaded into it."""
    fast.load_state_dict(reference.state_dict())
    grads_ref, grads_fast = gradients(reference, inputs, targets), gradients(fast, inputs, targets)
    total = math.sqrt(sum(g.square().sum().item() for g in grads_ref.values()))
    diff = math.sqrt(sum((grads_fast[n] - g).square().sum().item() for n, g in grads_ref.items()))
    floor = 1e-3 * total / math.sqrt(len(grads_ref))
    tensor, name = max(((grads_fast[n] - g).norm().item() / max(g.norm().item(), floor, 1e-30), n)
                       for n, g in grads_ref.items())
    return dict(grad=diff / max(total, 1e-30), tensor=tensor, name=name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=TRACKS, default="deep")
    parser.add_argument("--residual", required=True, help="the reference: entry file defining class Residual")
    parser.add_argument("--model", required=True, help="the fast model: entry file defining class GPT(model.GPT)")
    parser.add_argument("--draws", type=int, default=3, help="random weight draws to compare at (plus the init)")
    parser.add_argument("--batch", type=int, default=8, help="sequences per comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-data", action="store_true",
                        help="random tokens instead of FineWeb (only for testing the setup)")
    args = parser.parse_args()
    track = TRACKS[args.track]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 means fp32: TF32 matmuls would add ~1e-3 errors
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    residual_cls, _ = load_residual(args.residual)
    model_cls, _ = load_model(args.model)
    assert model_cls is not GPT, "--model must be a fast model, not the reference"
    print(f"torch {torch.__version__} on {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'}")
    print(f"args: {vars(args)}")

    shape = (VOCAB_SIZE, track.num_layers, track.model_dim, track.head_dim, residual_cls)
    torch.manual_seed(args.seed)
    reference, fast = GPT(*shape), model_cls(*shape)
    check_same_parameters(fast, reference)
    print(f"ok   parameters: the same {sum(1 for _ in reference.parameters())} names and shapes as the reference")
    init_backbone_(reference)
    if hasattr(reference.residual, "init_"):
        with torch.no_grad():
            reference.residual.init_(list(reference.sublayers))
    reference, fast = reference.to(device).float(), fast.to(device).float()

    n_tokens = args.batch * SEQ_LEN
    if args.synthetic_data:
        tokens = torch.randint(0, VOCAB_SIZE, ((args.draws + 1) * n_tokens + 1,))
    else:
        files = sorted(Path.cwd().glob(TRAIN_FILES))
        assert files, f"no data files match {TRAIN_FILES}; run: python data/cached_fineweb10B.py 1"
        tokens = _load_data_shard(files[0])

    ok = True
    for i in range(args.draws + 1):
        label = "init" if i == 0 else f"random weights {i}"
        if i > 0:
            randomize_(reference, args.seed + i)
        buf = tokens[i * n_tokens:(i + 1) * n_tokens + 1].to(device)
        inputs = buf[:-1].to(torch.int32).view(args.batch, SEQ_LEN)
        targets = buf[1:].to(torch.int64).view(args.batch, SEQ_LEN)
        e = compare(reference, fast, inputs, targets)
        good = e["grad"] <= GRAD_RTOL and e["tensor"] <= TENSOR_RTOL
        ok &= good
        print(f"{'ok' if good else 'FAIL':4} {label}: grad rel err {e['grad']:.1e} (tol {GRAD_RTOL:.0e}),"
              f" worst tensor {e['tensor']:.1e} (tol {TENSOR_RTOL:.0e}, {e['name']})")
    print(f"equivalence track:{args.track} residual:{args.residual} model:{args.model}: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
