# DepthDash

A time-to-target speedrun for deep-narrow Transformers where **only the residual connections may change**.
The format follows the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt), with timing
conventions from [Parameter Golf](https://github.com/openai/parameter-golf). The question comes from
[DepthBench](https://arxiv.org/abs/2609.32534).

DepthBench shows that with Pre-LN, making a fixed-size model deeper and narrower stops paying off.
Some residual designs (HC, Full AttnRes) keep improving. DepthDash turns that into a contest.
The data, backbone, init and optimizer are frozen. A submission changes the residual path, chooses its step
count and learning rate, and competes to reach the track's target val loss in the least training time on 8×H100.

## Tracks

Every track has nearly the same backbone (12·L·d² = 85–88M parameters, FineWeb, GPT-2 tokens, 524,288 tokens/step,
sequence length 1024). The tracks differ only in aspect ratio d/L.

| Track | Shape | Aspect ratio | Heads × dim | Backbone | Target val loss |
| - | - | - | - | - | - |
| `reference` (not scored) | L12 d768 | 64 | 6 × 128 | 84.93M | 3.28 |
| **Track 1: `deep`** | L28 d512 | 18.3 | 8 × 64 | 88.08M | TBD |
| **Track 2: `extreme_deep`** | L48 d384 | 8 | 6 × 64 | 84.93M | TBD |

Backbone counts the attention and MLP weight matrices that Muon trains, exactly 12·L·d². `deep` uses the round
shape L28 d512, which is 3.7% larger than the other two (L27 d512 would match exactly).
The train log prints `backbone_params`, `residual_params` and `total_params` for every run.

`reference` is upstream track 3's baseline shape. Pre-LN should reproduce upstream's 3.28 in 3250 steps there,
which makes it a regression check for the trainer.

### Calibrating the targets

A shallow target would reward residuals that converge fast early but stall. So each target is set where Pre-LN
ends a full 10-minute run:
1. Run `baselines/preln.py` with `--budget 600` (several seeds) to find the step count N that fits in 10 minutes.
2. Sweep Pre-LN's `--lr` at `--steps N`. The backbone lr was tuned for L12 d768, and an untuned baseline would
   make later records look better than they are.
3. Run Pre-LN at the best lr with `--steps N` (several seeds). Set the target to its mean final val loss,
   rounded up to a value it reaches reliably.
4. Record N and that lr as the track's default `steps` and `lr`. The tuned Pre-LN is record #1 at about 10:00.

Each track has its own target, so times compare within a track, not across tracks. To compare depths
directly, run every track for the same `--budget` and compare final losses.

## Quickstart

All runs use one node with 8× NVIDIA H100 SXM.

```bash
pip install -r requirements.txt
python data/cached_fineweb10B.py 40          # 4B training tokens + the validation shard
./run.sh --track deep --residual baselines/hc.py --steps 3000 --lr 0.025 --seed 0
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

The check runs both on one GPU in fp32 (TF32 off) from the same weights, at the init and at a few random draws,
and compares every gradient. A correct implementation differs by ~1e-6 there. The tolerances (1e-4 for the
gradients overall, 1e-2 per tensor) are far above that, so a correct fast implementation passes, while a wrong
scale, a missing transpose or a detached input fails. Differences far below the tolerances, such as a changed
epsilon, can slip through. That is fine: the rules are a gentlemen's agreement, and the check is there to catch
honest mistakes. For it to work, a fast implementation must also run in fp32, so write kernels for the input
dtype rather than hard-coding bf16.

[`baselines/fast/`](baselines/fast) has fast implementations of the baselines:
- `attnres.py` and `block_attnres.py` use DepthBench's fused Triton kernel;
- `mhc.py` uses Liger-Kernel;
- `preln.py` is a minimal example that folds each pre-norm gain into the next sublayer's input projections.

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

**Not allowed:** changing what the sublayers compute, e.g. value residuals or attention over earlier layers' keys
and values (MoDA). The reference `Residual` must be expressible with black-box sublayers.

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

## Leaderboard

### Track 1: deep (L28 d512)

| # | Time | Val loss (mean, n) | Steps | Residual | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |

### Track 2: extreme_deep (L48 d384)

| # | Time | Val loss (mean, n) | Steps | Residual | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |

## Layout

```
train_gpt.py     frozen: data loader, Muon, optimizer groups, init, warmup, timed training loop, validation, logging
model.py         frozen: RMSNorm, attention, MLP, GPT (embed -> residual -> head)
tracks.py        frozen: track shapes, calibrated step counts and targets
check_equivalence.py  frozen: checks that a fast model (--model) computes the same function as its reference
baselines/       reference residuals: the plain PyTorch math, one entry file each
baselines/fast/  fast models with kernels (--model): attnres and block_attnres (Triton kernel from
                 DepthBench), mhc (Liger-Kernel), preln (a minimal example)
records/         submissions per track
data/            FineWeb 10B (GPT-2 tokens) downloader
```

## Acknowledgements

- **Trainer and scoring:** descends from modded-nanogpt's track 3 (`train_gpt_simple.py`); time-to-target
  scoring follows its main track.
- **Timing conventions:** the off-the-clock compile warmup and the fixed-time `--budget` mode follow Parameter Golf.
- **HC/mHC:** the implementations come from [nanogpt-lab](https://github.com/aryagxr/nanogpt-lab).
- **Upstream history:** modded-nanogpt's history, including its tracks 1–3 and all their records, remains in
  this repository's git history.
