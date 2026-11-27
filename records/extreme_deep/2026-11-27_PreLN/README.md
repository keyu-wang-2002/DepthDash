# Track 2 `extreme_deep`: Pre-LN baseline

The calibration run that sets the `extreme_deep` target: Pre-LN, the plain reference math, with no fast model.

- **Residual:** [`baselines/preln.py`](../../../baselines/preln.py), unchanged (`h <- h + F(RMSNorm(h))`)
- **Steps / lr:** `--steps 2150 --lr 0.025`
- **Time:** 599.3 s mean `train_time` (598.59–599.80 s), on 8× H100 SXM
- **Val loss:** 3.42905 mean over 5 seeds (sd 0.00045)
- **Target:** 3.430, met at p ≈ 0.005 (one-sided t-test, n = 5, t = −4.69, df = 4)

```bash
./run.sh --track extreme_deep --residual baselines/preln.py --steps 2150 --lr 0.025 --seed S --val-every 250 --compile sublayer
```

`--compile sublayer` and `--val-every` change only execution and logging, not the math.

| Seed | Final val loss | `train_time` | Log |
| - | - | - | - |
| 0 | 3.42888 | 599.114 s | [log](extreme_deep-preln-98904763-adee-46a8-8671-044a31ed4770.txt) |
| 1 | 3.42837 | 598.586 s | [log](extreme_deep-preln-2867a667-2d55-4097-9b78-3eb746c919c2.txt) |
| 2 | 3.42928 | 599.464 s | [log](extreme_deep-preln-153150ab-75df-468b-99cf-5209abab7be6.txt) |
| 3 | 3.42956 | 599.638 s | [log](extreme_deep-preln-edaf4082-4467-4a15-9bcf-9ce181223ea4.txt) |
| 4 | 3.42915 | 599.797 s | [log](extreme_deep-preln-21d73960-d8c6-4f99-8c86-14f9dff3bf44.txt) |
| **Mean** | **3.42905** | **599.32 s** | |

## How the steps and lr were chosen

- **Steps:** 2150 steps take about 600 s (`CALIBRATION_BUDGET` in `tracks.py`). Pre-LN runs at a steady
  278–279 ms/step on `extreme_deep`, against 209.8 ms/step on `deep` and 162 ms/step on `reference`.
- **lr:** a sweep of the Muon lr at 2150 steps (seed 0) found 0.025 best, in the middle of the range:

  | lr | 0.015 | 0.02 | **0.025** | 0.035 | 0.05 |
  | - | - | - | - | - | - |
  | val loss | 3.43358 | 3.43037 | **3.42888** | 3.43409 | 3.44264 |

- **Target:** 3.430 is just above the Pre-LN mean, so Pre-LN itself clears it at p < 0.01 and stands as record #1,
  as for `deep` (Pre-LN 3.3348, target 3.336) and `reference` (Pre-LN 3.278, target 3.28).

The logs come from a checkout before Track 3 was added, so their embedded `tracks.py` has no `open` entry and
still says `target=None` for `extreme_deep`. `train_gpt.py`, `model.py` and `baselines/preln.py` are identical to
the current ones.
