"""DepthDash: time-to-target speedrun for deep-narrow GPTs where only the residual connections may change.

Launch: torchrun --standalone --nproc_per_node=8 train_gpt.py --track deep --residual baselines/hc.py --steps 3000

This file, model.py, tracks.py and check_equivalence.py are frozen (data, backbone, init, optimizer, schedule,
evaluation). A submission is a residual module (an entry file defining class Residual, optionally with helper
modules and custom kernels next to it) plus its --steps and --lr. See baselines/preln.py for the interface.
The Residual is the reference: it defines the math. A submission may also bring a fast model (--model), a
`class GPT(model.GPT)` that computes the same function any way it likes, verified by check_equivalence.py.
The score is the training time (compile warmup and validation off the clock) of a run whose final val loss
reaches the track's target, as in modded-nanogpt. --budget instead trains for a fixed time, as in
openai/parameter-golf; that is for depth-scaling diagnostics and target calibration, not for records.
Descends from modded-nanogpt's track 3 trainer (records/track_3_optimization/train_gpt_simple.py upstream).
"""
import argparse
import copy
import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim import AdamW

from model import GPT
from tracks import BATCH_TOKENS, SEQ_LEN, TRACKS, TRAIN_FILES, VAL_FILES, VAL_TOKENS, VOCAB_SIZE

ROOT = Path(__file__).resolve().parent


########################################
#              Dataloader              #
########################################

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=SEQ_LEN):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert files, f"no data files match {filename_pattern}; run: python data/cached_fineweb10B.py 40"
    assert batch_size % dist.get_world_size() == 0
    local_batch_size = batch_size // dist.get_world_size()
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            next_file = next(file_iter, None)
            if next_file is None:
                raise RuntimeError(f"ran out of the {len(files)} training shards; download more with "
                                   "python data/cached_fineweb10B.py N")
            tokens, pos = _load_data_shard(next_file), 0
        buf = tokens[pos + dist.get_rank() * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#              Optimizer               #
########################################

def zeropower_via_newtonschulz5(G: Tensor) -> Tensor:
    assert G.ndim >= 2
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations, not optimizing for wallclock speed
    a, b, c = 2, -1.5, 0.5
    for _ in range(12):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def muon_update(grad, momentum, mu=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - mu)
    update = grad.lerp_(momentum, mu) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, mu=0.95):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        defaults = dict(lr=lr, weight_decay=weight_decay, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum"], mu=group["mu"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank])


# Frozen hyperparameters (upstream track 3 baseline). A submission tunes only --steps and --lr (the Muon lr of
# the sublayers' matrices), plus AdamW settings for the residual's own parameters (optim_groups, defaulting to
# the scalar group's). The schedule is fixed: no warmup, linear decay to 0 over the last COOLDOWN_FRAC.
EMBED_LR, HEAD_LR, SCALAR_LR, ADAM_WD = 0.7, 0.004, 0.015, 0.001
MUON_WD = 0.05
COOLDOWN_FRAC = 0.7

def lr_schedule(progress: float) -> float:
    """Stable, then linear decay to 0 over the last COOLDOWN_FRAC (upstream track 3's schedule).
    progress = step / steps (or elapsed / budget)."""
    return 1.0 if progress < 1 - COOLDOWN_FRAC else max(1 - progress, 0.0) / COOLDOWN_FRAC

def adam_param_groups(model: GPT) -> list:
    """The AdamW groups: embed, head, backbone scalars, then the residual's own groups."""
    residual = model.residual
    residual_ids = {id(p) for p in residual.parameters()}
    if hasattr(residual, "optim_groups"):
        residual_groups = residual.optim_groups()
    else:
        residual_groups = [dict(params=list(residual.parameters()))]
    residual_groups = [{"lr": SCALAR_LR, "weight_decay": ADAM_WD, **g, "params": list(g["params"])}
                       for g in residual_groups]
    backbone_scalars = [p for p in model.parameters() if p.ndim < 2 and id(p) not in residual_ids]
    groups = [dict(params=[model.embed.weight], lr=EMBED_LR),
              dict(params=[model.proj.weight], lr=HEAD_LR),
              dict(params=backbone_scalars, lr=SCALAR_LR),
              *residual_groups]
    return [g for g in groups if g["params"]]

def muon_params(model: GPT) -> list:
    return [p for p in model.sublayers.parameters() if p.ndim >= 2]

def check_partition(model: GPT, groups: list):
    """Every parameter is optimized exactly once (also catches a residual that registers sublayers)."""
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)), "a parameter is in more than one optimizer group"
    assert set(ids) == {id(p) for p in model.parameters()}, "some parameters are not optimized"

def build_optimizers(model: GPT, muon_lr: float):
    adam_groups = adam_param_groups(model)
    check_partition(model, adam_groups + [dict(params=muon_params(model))])
    adam = AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=ADAM_WD, fused=True)
    muon = Muon(muon_params(model), lr=muon_lr, weight_decay=MUON_WD)
    optimizers = [adam, muon]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return optimizers

def set_lr(optimizers, progress: float):
    eta = lr_schedule(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * eta


########################################
#           Model construction         #
########################################

KERNEL_SOURCE_SUFFIXES = (".cu", ".cuh", ".cpp", ".h")

def load_entry(path: str, name: str):
    """Import a submission's entry file, which defines class `name`. Its directory goes on sys.path, so a
    submission can keep helpers and custom kernels in sibling modules. Returns the class and every source file
    of the submission (the entry, the modules it imported from its directory, and any C++/CUDA sources there)
    for the log."""
    entry = Path(path).resolve()
    directory = entry.parent
    sys.path.insert(0, str(directory))
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location(name.lower(), entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, name), f"{path} must define class {name}"
    imported = {Path(m.__file__).resolve() for name, m in sys.modules.items()
                if name not in before and getattr(m, "__file__", None)}
    files = [entry] + sorted(f for f in imported if f.is_relative_to(directory) and f != entry)
    files += sorted(f for f in directory.rglob("*") if f.suffix in KERNEL_SOURCE_SUFFIXES)
    return getattr(module, name), files

def load_residual(path: str):
    return load_entry(path, "Residual")

def load_model(path):
    """The model class: the frozen GPT, or a submission's fast `class GPT(model.GPT)`."""
    if path is None:
        return GPT, []
    model_cls, files = load_entry(path, "GPT")
    assert issubclass(model_cls, GPT), f"{path} must define class GPT as a subclass of model.GPT"
    return model_cls, files

def check_same_parameters(model: GPT, reference: GPT):
    """A fast model keeps the reference's parameters (names and shapes), so init and optimizer groups match."""
    ours = {n: tuple(p.shape) for n, p in model.named_parameters()}
    theirs = {n: tuple(p.shape) for n, p in reference.named_parameters()}
    assert ours == theirs, ("the fast model's parameters differ from the reference's: "
                            f"extra {sorted(ours.keys() - theirs.keys())}, missing {sorted(theirs.keys() - ours.keys())}, "
                            f"reshaped {sorted(n for n in ours.keys() & theirs.keys() if ours[n] != theirs[n])}")

@torch.no_grad()
def init_backbone_(model: GPT):
    """Frozen init of everything but the residual (which initializes itself in __init__)."""
    for name, p in model.named_parameters():
        if name.startswith("residual."):
            continue
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            elif "embed" in name:
                w.normal_()  # default torch init
            else:
                w.normal_(std=0.33**0.5 / w.size(-1)**0.5)  # default torch init
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.fill_(1)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

def compile_model(model: GPT, mode: str):
    if mode == "model":
        model.compile(dynamic=False)
    elif mode == "sublayer":  # for residuals whose whole-model graph is too large to compile
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 2 * len(model.sublayers))
        for sublayer in model.sublayers:
            sublayer.compile(dynamic=False)
        model.head_loss = torch.compile(model.head_loss, dynamic=False)
    else:
        assert mode == "none", mode


########################################
#          Training & evaluation       #
########################################

def train_step(model: GPT, optimizers, inputs: Tensor, targets: Tensor, mbs: int):
    # accumulate across microbatches (fewer than 8 gpus, or a small --microbatch)
    for i in range(len(inputs) // mbs):
        model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

@torch.no_grad()
def evaluate(model: GPT, val_inputs: Tensor, val_targets: Tensor, mbs: int) -> float:
    model.eval()
    val_loss = 0
    for i in range(len(val_inputs) // mbs):
        val_loss += model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
    model.train()
    return val_loss.item() / VAL_TOKENS


########################################
#                 Main                 #
########################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=TRACKS, default="deep")
    parser.add_argument("--residual", default="baselines/preln.py", help="entry file defining class Residual")
    parser.add_argument("--model", help="optional fast model: entry file defining class GPT(model.GPT) that computes"
                        " the same function as the reference; must pass check_equivalence.py")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", type=int, help="seed for trial 0; trial i uses seed + i")
    parser.add_argument("--val-every", type=int, default=0, help="validate every N steps (off the clock); 0 = final only")
    parser.add_argument("--steps", type=int, help="training steps, part of the submission (default: the track's)")
    parser.add_argument("--lr", type=float, help="Muon lr of the sublayers' matrices, part of the submission"
                        " (default: the track's)")
    # Diagnostics only (not a record): train for a fixed time with a time-based lr schedule.
    parser.add_argument("--budget", type=float, help="train for this many seconds instead of --steps")
    # Execution only: none of these change the math.
    parser.add_argument("--compile-warmup", type=int, default=20,
                        help="off-the-clock steps to compile, after which weights and optimizer are reset")
    parser.add_argument("--microbatch", type=int, help="sequences per forward per GPU (default: the track's)")
    parser.add_argument("--checkpoint", action="store_true", help="activation checkpointing of each sublayer")
    parser.add_argument("--compile", choices=["auto", "model", "sublayer", "none"], default="auto",
                        help="auto = the fast model's or the residual's compile_mode, else 'model'")
    return parser.parse_args()

def main():
    args = parse_args()
    track = TRACKS[args.track]
    budget = args.budget
    steps = None if budget else (args.steps or track.steps)
    assert (budget or 0) > 0 or steps, f"track {args.track} has no calibrated step count yet: pass --steps or --budget"
    assert not (args.budget and args.steps), "--steps and --budget are exclusive"
    record_run = budget is None
    lr = args.lr or track.lr
    mbs = args.microbatch or track.microbatch
    residual_cls, residual_files = load_residual(args.residual)
    model_cls, model_files = load_model(args.model)
    sources = {name: (ROOT / name).read_text() for name in ("train_gpt.py", "model.py", "tracks.py")}
    sources.update({os.path.relpath(f): f.read_text() for f in residual_files + model_files})

    # torchrun sets these env variables
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    world_size = dist.get_world_size()
    # this code can be run equivalently with 1, 2, 4, or 8 gpus; record runs use 8xH100.
    assert 8 % world_size == 0
    sequences_per_rank = BATCH_TOKENS // world_size // SEQ_LEN
    assert sequences_per_rank % mbs == 0, f"--microbatch must divide {sequences_per_rank}"

    # logging setup
    if dist.get_rank() == 0:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.track}-{Path(args.residual).stem}-{uuid.uuid4()}.txt"
        print(logfile)
    def print0(s, console=False, log=True):
        if dist.get_rank() == 0:
            if console:
                print(s)
            if log:
                with open(logfile, "a") as f:
                    print(s, file=f)

    # we begin by logging the source of this run
    for name, source in sources.items():
        print0(f"{'=' * 40} {name} {'=' * 40}\n{source}")
    print0("=" * 100)
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}"
           + f" on {torch.cuda.get_device_name(device)} with world_size {world_size}")
    print0(f"args: {vars(args)}")
    print0("=" * 100)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    shape = (VOCAB_SIZE, track.num_layers, track.model_dim, track.head_dim, residual_cls)
    if model_cls is not GPT:
        check_same_parameters(model_cls(*shape), GPT(*shape))
    model = model_cls(*shape).cuda()
    model.activation_checkpointing = args.checkpoint
    residual_init_state = copy.deepcopy(model.residual.state_dict())
    default_compile = getattr(model_cls, "compile_mode", None) or getattr(residual_cls, "compile_mode", "model")
    compile_mode = default_compile if args.compile == "auto" else args.compile
    compile_model(model, compile_mode)

    residual_params = sum(p.numel() for p in model.residual.parameters())
    backbone_params = sum(p.numel() for p in muon_params(model))
    limit = f"steps:{steps}" if steps else f"budget:{budget:.0f}s"
    model_name = args.model or "model.py"
    print0(f"track:{args.track} model:{model_name} layers:{track.num_layers} dim:{track.model_dim} aspect_ratio:{track.aspect_ratio:.2f}"
           + f" backbone_params:{backbone_params} residual_params:{residual_params}"
           + f" total_params:{sum(p.numel() for p in model.parameters())} compile:{compile_mode} {limit}"
           + f" lr:{lr}"
           + f" record_run:{record_run}", console=True)

    val_inputs, val_targets = next(distributed_data_generator(VAL_FILES, VAL_TOKENS))
    assert len(val_inputs) % mbs == 0

    for trial in range(args.trials):

        ########################################
        #       Init & compile warmup          #
        ########################################

        if args.seed is not None:
            torch.manual_seed(args.seed + trial)
        init_backbone_(model)
        model.residual.load_state_dict(residual_init_state)
        if hasattr(model.residual, "init_"):
            with torch.no_grad():
                model.residual.init_(list(model.sublayers))
        for p in model.parameters():
            dist.broadcast(p.detach(), 0)

        # Warmup compiles the training step off the clock; then the weights, optimizer state and data
        # stream are reset, so the timed run starts from the true init (as in parameter-golf).
        if args.compile_warmup > 0:
            initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            optimizers = build_optimizers(model, lr)
            set_lr(optimizers, 0.0)
            warmup_loader = distributed_data_generator(TRAIN_FILES, BATCH_TOKENS)
            for _ in range(args.compile_warmup):
                train_step(model, optimizers, *next(warmup_loader), mbs)
            model.load_state_dict(initial_state)
            del initial_state, warmup_loader
        optimizers = build_optimizers(model, lr)
        train_loader = distributed_data_generator(TRAIN_FILES, BATCH_TOKENS)
        torch.cuda.reset_peak_memory_stats(device)


        ########################################
        #        Training and Validation       #
        ########################################

        def shared_clock(local_seconds: float) -> float:
            # Every rank uses the max, so in --budget mode all ranks see the same lr and stop on the same
            # step. The .item() also waits for the GPU, so the clock measures finished work.
            seconds = torch.tensor(local_seconds, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            return seconds.item()

        training_time = 0.0
        step = 0
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while True:
            if budget:  # a per-step sync, only paid in the fixed-time diagnostic mode
                progress = shared_clock(training_time + time.perf_counter() - t0) / budget
            else:
                progress = step / steps
            last_step = progress >= 1

            if last_step or (args.val_every and step > 0 and step % args.val_every == 0):
                # stop the clock
                torch.cuda.synchronize()
                training_time = shared_clock(training_time + time.perf_counter() - t0)
                val_loss = evaluate(model, val_inputs, val_targets, mbs)
                print0(f"step:{step} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
                       + f" step_avg:{1000 * training_time / max(step, 1):.2f}ms", console=True)
                if last_step:
                    break
                # start the clock again
                dist.barrier()
                t0 = time.perf_counter()

            set_lr(optimizers, progress)
            train_step(model, optimizers, *next(train_loader), mbs)
            step += 1
            approx_time = training_time + time.perf_counter() - t0
            print0(f"step:{step} train_time:{approx_time:.3f}s step_avg:{1000 * approx_time / step:.2f}ms",
                   console=True, log=False)

        peak_mem = torch.tensor(torch.cuda.max_memory_allocated(device), dtype=torch.float64, device=device)
        dist.all_reduce(peak_mem, op=dist.ReduceOp.MAX)
        reached = "n/a" if track.target is None else val_loss <= track.target
        print0(f"result track:{args.track} residual:{args.residual} model:{model_name} trial:{trial} seed:{args.seed}"
               + f" val_loss:{val_loss:.5f} target:{track.target} reached:{reached} train_time:{training_time:.3f}s"
               + f" steps:{step} {limit} record_run:{record_run} peak_mem:{peak_mem.item() / 2**30:.1f}GiB",
               console=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
