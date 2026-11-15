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

| Track | L × d | Aspect ratio | Heads × dim | Backbone | Target val loss |
| - | - | - | - | - | - |
| `reference` (not scored) | 12 × 768 | 64 | 6 × 128 | 84.93M | 3.28 |
| **`deep`** | 28 × 512 | 18.3 | 8 × 64 | 88.08M | TBD |
| **`extreme_deep`** | 48 × 384 | 8 | 6 × 64 | 84.93M | TBD |

Backbone counts the attention and MLP weight matrices that Muon trains, exactly 12·L·d². `deep` uses the round
shape 28 × 512, which is 3.7% larger than the other two (27 × 512 would match exactly).
The train log prints `backbone_params`, `residual_params` and `total_params` for every run.

`reference` is upstream track 3's baseline shape. Pre-LN should reproduce upstream's 3.28 in 3250 steps there,
which makes it a regression check for the trainer.

### Calibrating the targets

A shallow target would reward residuals that converge fast early but stall. So each target is set where Pre-LN
ends a full 10-minute run:
1. Run `baselines/preln.py` with `--budget 600` (several seeds) to find the step count N that fits in 10 minutes.
2. Sweep Pre-LN's `--lr` at `--steps N`. The backbone lr was tuned for 12 × 768, and an untuned baseline would
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
./run.sh --track reference --residual baselines/preln.py --steps 3250 --lr 0.025 --seed 0   # regression check: ~3.28
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

1. **Frozen:** `train_gpt.py`, `model.py`, `tracks.py`. A submission is a residual module plus its `--steps`.
2. **Only the residual path.** The residual treats sublayers as black boxes:
   - it calls each sublayer exactly once per forward;
   - it never reads their weights or internals during the forward;
   - it only operates within each token position (no mixing across the sequence).
3. **Init.** `init_` may rescale sublayer weights by constants (e.g. a depth-dependent scale). Nothing else.
4. **Kernels.** Custom kernels and fusions are welcome for the residual's own computation (e.g. a fused HC
   read/write); they cut the time directly. Keep a plain PyTorch path to check them against.
   Kernels may not reach into the sublayers.
5. **Hyperparameters.** Allowed: `--steps`, `--lr`, and AdamW lr and weight decay for the residual's own
   parameters. Everything else (schedule shape, batch size, betas, embedding/head/scalar lrs, data) is fixed.
   Tuning across runs is fine; brute-forcing seeds is not.
6. **Fixed endpoint.** The number of steps and the lr are set before the run starts. Nothing may depend
   on validation loss (no early stopping, no schedule changes mid-run), and the training run never reads
   validation data.
7. **Score:** the training time printed in the `result` line, on 8× H100 SXM. Compile warmup and validation
   are off the clock. As in modded-nanogpt, the run logs must show at p < 0.01 that the mean final val loss is
   at or below the track's target. Usually that takes several seeds.

To submit, add `records/<track>/<YYYY-MM-DD>_<Name>/` with:
- the residual module (entry file, helpers, kernels);
- the logs of all seeds, run with `--residual records/<track>/<YYYY-MM-DD>_<Name>/<entry>.py --steps N --lr LR`;
- a short README with the time, the mean val loss and the p-value against the target.

The log embeds the entry file, every module it imported from its directory, and any `.cu/.cuh/.cpp/.h` sources
there.

## Leaderboard

### deep (28 × 512)

| # | Time | Val loss (mean, n) | Steps | Residual | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |

### extreme_deep (48 × 384)

| # | Time | Val loss (mean, n) | Steps | Residual | Date | Log | Contributors |
| - | - | - | - | - | - | - | - |

## Layout

```
train_gpt.py     frozen: data loader, Muon, optimizer groups, init, warmup, timed training loop, validation, logging
model.py         frozen: RMSNorm, attention, MLP, GPT (embed -> residual -> head)
tracks.py        frozen: track shapes, calibrated step counts and targets
baselines/       reference residuals, one self-contained file each
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
