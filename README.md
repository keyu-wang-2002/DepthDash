# DepthDash


This repository hosts the *DepthDash* speedrun, in which we (collaboratively|competitively) search for the fastest residual connection design to use 8 NVIDIA H100 GPUs to train a *deep-narrow* language model that attains targeted cross-entropy loss on the FineWeb validation set.

[DepthBench](https://arxiv.org/abs/2609.32534) shows that depth can serve as a scaling axis if your residuals let it be. So we build DepthDash on the [Modded-NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), with timing conventions from [Parameter Golf](https://github.com/openai/parameter-golf), but with a much more deeper and narrower backbone.

The data, backbone, init and optimizer are frozen. A submission changes only the residual path, with system and kernel optimizations allowed, and picks its step count and learning rate.

## Tracks

Every track has nearly the same backbone (12·L·d² = 85–88M parameters, FineWeb, GPT-2 tokens, 524,288 tokens/step,
sequence length 1024). The tracks differ only in aspect ratio d/L. `reference` is the setting used in Modded-NanoGPT.
Tracks 1 and 2 change only the residual connections. Track 3 keeps the deep backbone and target, but otherwise,
like Modded-NanoGPT's main track, anything goes within a 5% parameter budget, and yes, optimizer co-design is allowed (see [Rules](#rules)).

| Track | Shape | Aspect ratio | Heads × dim | Backbone | Total | Target val loss |
| - | - | - | - | - | - | - |
| `reference` (not scored) | L12 d768 | 64 | 6 × 128 | 84.93M | 162.35M | 3.28 |
| **Track 1: `deep`** | L28 d512 | 18.3 | 8 × 64 | 88.08M | 139.80M | 3.336 |
| **Track 2: `extreme_deep`** | L48 d384 | 8 | 6 × 64 | 84.93M | 123.82M | TBD |
| **Track 3: `open`** | L28 d512 | 18.3 | 8 × 64 | 88.08M | 139.80M | 3.336 |


## Quickstart

All runs use one node with 8× NVIDIA H100 SXM.

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 40          # 4B training tokens + the validation shard

# a simple example: Pre-LN, written as plain PyTorch math
./run.sh --track deep --residual baselines/preln.py --steps 3000 --lr 0.025 --seed 0
# --model overrides the math version with a rewritten model file, for kernel and system speedups
./run.sh --track deep --residual baselines/preln.py --model baselines/fast/preln.py --steps 3000 --lr 0.025 --seed 0
```

A run proceeds like this:
1. Compile warmup: 20 training steps, after which weights, optimizer state and the data stream are reset to the
   true init. This is off the clock.
2. `--steps` training steps at `--lr` (see [Steps and learning rate](#steps-and-learning-rate)).
3. A final validation on the fixed 10.5M-token FineWeb validation set. This is also off the clock.

Each run writes `logs/<track>-<residual>-<uuid>.txt`. It contains the full source of the run and ends with a
`result ... val_loss ... reached ... train_time ...` line.

Flags:
- `--val-every N`: intermediate validation, off the clock.
- `--microbatch N`: sequences per forward per GPU. Lower it if you run out of memory.
- `--checkpoint`: activation checkpointing of every sublayer.
- `--compile {auto,model,sublayer,none}`
- `--model PATH`: a fast model that computes the same function as the residual (see [Fast models](#fast-models)).
- `--budget SECONDS`: train for a fixed time with a time-based lr schedule, as in Parameter Golf. This is for
  target calibration and depth-scaling diagnostics; such runs are marked `record_run:False`.

`--microbatch`, `--checkpoint` and `--compile` change speed and memory, not the math.

## The residual interface

A submission is a residual module. Its entry file defines `class Residual(nn.Module)` and may import helper
modules and custom kernels (Triton, CUDA) kept in the same directory:

```python
class Residual(nn.Module):
    def __init__(self, dim, num_sublayers): ...
    def forward(self, x0, sublayers):  # x0: normalized embedding (B,T,D)
        ...                            # sublayers: [attn_0, mlp_0, attn_1, ...], each a pure F(x), no norm, no add
        return h                       # (B,T,D) -> frozen final RMSNorm -> lm_head
```

Pre-LN is just `for norm, f in zip(self.norms, sublayers): x = x + f(norm(x))`.

Optional hooks (also documented in [`baselines/preln.py`](baselines/preln.py)):
- `init_(sublayers)`: rescaling of sublayer weights at init
- `compile_mode`: `"model"` (default) or `"sublayer"`, for residuals whose whole-model graph is too large to compile

### Fast models

A submission has a reference implementation and, optionally, a fast implementation.

- **Reference implementation:** the `Residual`, written in plain PyTorch with the sublayers as black boxes.
  It defines the math.
- **Fast implementation** (`--model`): an entry file defining `class GPT(model.GPT)` that overrides `forward`.
  It may compute the function any way it likes. For example, it may fuse one sublayer's write with the next
  one's read, fold a pre-norm into the QKV matmul, use custom Triton/CUDA kernels, or use libraries such as
  Liger-Kernel.

**The fast implementation must be mathematically identical to the reference.** Only rounding may differ: no
approximations, no changed constants, no lower-precision tricks such as FP8. It also keeps the reference's
parameters (names and shapes), so init and optimizer groups stay the same.

We provide [`check_equivalence.py`](check_equivalence.py) as a rough check, not a proof:

```bash
python check_equivalence.py --track deep --residual <dir>/residual.py --model <dir>/fast_model.py
./run.sh --track deep --residual <dir>/residual.py --model <dir>/fast_model.py --steps N --lr LR --seed 0
```


### Steps and learning rate

As in modded-nanogpt's track 3, the schedule is fixed: no warmup, constant lr, then linear decay to 0 over
the last 70% of the steps. A submission tunes two flags, plus the lr of the residual's own parameters:

| Knob | Where | Default |
| - | - | - |
| Steps N | `--steps N` | the track's calibrated `steps` (pass it while the track is uncalibrated) |
| Muon lr of the sublayers' matrices | `--lr LR` | the track's calibrated `lr` (0.025 until calibrated) |
| lr / weight decay of the residual's own parameters | method `optim_groups()` in the residual file | lr 0.015, weight decay 0.001 |

The embedding (0.7), head (0.004) and backbone scalar (0.015) lrs are frozen; the schedule scales them too.
The residual's own lrs live in its file because they are part of its design (e.g. HC's routing at 1e-3).

```bash
./run.sh --track deep --residual records/deep/2026-12-01_MyResidual/residual.py --steps 2800 --lr 0.03 --seed 0
```

```python
# records/deep/2026-12-01_MyResidual/residual.py
class Residual(nn.Module):
    def optim_groups(self):   # optional: AdamW groups for this module's own parameters
        return [dict(params=self.norms.parameters()),                      # defaults: lr 0.015, wd 0.001
                dict(params=self.routing.parameters(), lr=1e-3, weight_decay=0.1)]
    ...
```

The run log records the full command line and prints the `steps` and `lr` actually used.

## Rules

### Tracks 1 and 2: residual connections only

**Frozen:** `train_gpt.py`, `model.py`, `tracks.py`, `check_equivalence.py`. That covers the data, the batch size
(524,288 tokens/step), the backbone, its init, the optimizers, the lr schedule and the evaluation.

**You may change:**
- **The residual.** Your `Residual` module decides the norms, how sublayer outputs are combined and what reaches
  the head. Its `init_` may rescale sublayer weights by constants.
- **The implementation.** You may use custom Triton/CUDA kernels and fusions, inside the residual or in a fast
  model (`--model`) that crosses the sublayer boundary, as long as it passes `check_equivalence.py`.
- **Hyperparameters.** You may set `--steps`, `--lr` (the Muon lr of the sublayer matrices), and the AdamW lr
  and weight decay of the residual's own parameters (`optim_groups`).
- **Execution.** `--microbatch`, `--checkpoint` and `--compile` change speed and memory, not the math.

**Not supported yet:** methods that change what the sublayers compute, e.g. value residuals or attention over
earlier layers' keys and values (MoDA). For now, the reference `Residual` must be expressible with black-box
sublayers. Such methods can compete in Track 3.

**Constraints:**
- In the reference `Residual`, sublayers are black boxes. It calls each one exactly once per forward, in order,
  and never reads their weights or internals. A fast model may reach in, but only to compute the same function.
- The residual works per token. It may not mix positions within a sequence or sequences within a batch.
- Steps and lr are fixed before the run. Nothing may depend on val loss, and training never reads validation data.
- Tune as much as you like across runs. Do not cherry-pick seeds.

**Score:** the `train_time` in the `result` line, on 8× H100 SXM. Compile warmup and validation are off the
clock. The logs must show at p < 0.01 that the mean final val loss is at or below the track's target. That
usually takes several seeds.

To submit, add `records/<track>/<YYYY-MM-DD>_<Name>/` with:
- the residual module (entry file, helpers, kernels), and the fast model if you use one;
- the logs of all seeds, run with `--residual records/<track>/<YYYY-MM-DD>_<Name>/<entry>.py --steps N --lr LR`
  (plus `--model ...`);
- with a fast model, the output of `check_equivalence.py`;
- a short README with the time, the mean val loss and the p-value against the target.

The log embeds the entry file, every module it imported from its directory, and any `.cu/.cuh/.cpp/.h` sources
there.

### Track 3: open

Like Modded-NanoGPT's main track: train the `deep` backbone to the `deep` target as fast as you can, and change
anything else you like. That includes the residual path and anything added around the backbone (value residuals,
MoDA, gates, extra embeddings), how the attention and MLP compute (norms, activations, attention patterns), the
init, the optimizer and its schedule, the batch size and sequence length, kernels and precision (e.g. FP8).

**Keep:**
- **The backbone:** as in `deep`: 28 blocks with a residual stream width of 512, each with one attention
  (8 heads × 64, matrices 512 × 512 for Q, K, V and the output) and one MLP (512 → 2048 → 512).
- **The parameter budget:** at most 5% more parameters in total than `deep` with Pre-LN, i.e.
  ≤ 146,790,739 (139,800,704 + 5%), counting the embedding, the head and everything you add.
- **Data and evaluation:** the same train and validation token streams, and the same val loss, i.e.
  cross-entropy on the 10,485,760 FineWeb validation tokens.
- **Timing:** 8× H100 SXM. Compile warmup and validation stay off the clock, and the run ends with the `result`
  line.
- **Fair play:** nothing may depend on val loss, and training never reads validation data. Do not cherry-pick
  seeds.

**Score:** as in Tracks 1 and 2. It is the `train_time` of runs whose mean final val loss is at or below the
`deep` target at p < 0.01. Since the backbone and the target are the same, Track 3 times compare directly
with Track 1 times.

To submit, copy `train_gpt.py`, `model.py`, `tracks.py` (and whatever else you need) into
`records/open/<YYYY-MM-DD>_<Name>/`, change them as you like, and run from the repository root:

```bash
torchrun --standalone --nproc_per_node=8 records/open/<YYYY-MM-DD>_<Name>/train_gpt.py --track open --steps N
```

Add the logs of all seeds and a short README with the time, the mean val loss and the p-value against the
target. The log embeds your `train_gpt.py`, `model.py` and `tracks.py`; make it log any other files you add too.

## Leaderboard

### Track 1: deep (L28 d512)

| # | Time | Val loss (mean, n) | Steps | Residual | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |
| 1 | 596.9 s | 3.3348 (5) | 2850 | Pre-LN | 2026-11-27 | [log](records/deep/2026-11-27_PreLN/) | [keyu-wang-2002](https://github.com/keyu-wang-2002) |

### Track 2: extreme_deep (L48 d384)

| # | Time | Val loss (mean, n) | Steps | Residual | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |

### Track 3: open (L28 d512)

| # | Time | Val loss (mean, n) | Steps | Description | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |

## Layout

```text
DepthDash/
├── train_gpt.py               # frozen: data loader, Muon, optimizer groups, init, warmup, timed loop, validation, logging
├── model.py                   # frozen: RMSNorm, attention, MLP, GPT (embed -> residual -> head)
├── tracks.py                  # frozen: track shapes, step counts and targets
├── check_equivalence.py       # frozen: checks that a fast model (--model) computes the same function as its reference
├── baselines/                 # reference residuals: the plain PyTorch math, one entry file each
│   └── fast/                  # fast models with kernels (--model)
├── records/                   # submissions, per track
│   ├── deep/
│   ├── extreme_deep/
│   └── open/
├── data/                      # FineWeb 10B (GPT-2 tokens) downloader
├── run.sh                     # torchrun launcher for 8 GPUs
└── requirements.txt
```

## Acknowledgements

- **Trainer and scoring:** descends from modded-nanogpt's track 3 (`train_gpt_simple.py`); time-to-target
  scoring follows its main track.
- **Timing conventions:** the off-the-clock compile warmup and the fixed-time `--budget` mode follow Parameter Golf.
