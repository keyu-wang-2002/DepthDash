# Track 1 `deep`: Pre-LN baseline

The calibration run that sets the `deep` target: Pre-LN, the plain reference math, with no fast model.

- **Residual:** [`baselines/preln.py`](../../../baselines/preln.py), unchanged (`h <- h + F(RMSNorm(h))`)
- **Steps / lr:** `--steps 2850 --lr 0.025`
- **Time:** 596.9 s mean `train_time` (595.98–597.74 s), on 8× H100 SXM
- **Val loss:** 3.33480 mean over 5 seeds (sd 0.00065)
- **Target:** 3.336, met at p ≈ 0.007 (one-sided t-test, n = 5, t = −4.17, df = 4)

```bash
./run.sh --track deep --residual baselines/preln.py --steps 2850 --lr 0.025 --seed S --val-every 250 --compile sublayer
```

`--compile sublayer` and `--val-every` change only execution and logging, not the math.

| Seed | Final val loss | `train_time` | Log |
| - | - | - | - |
| 0 | 3.33398 | 595.980 s | [log](deep-preln-41e98369-d1ba-40e7-a564-2fb5682e9cd5.txt) |
| 1 | 3.33530 | 597.097 s | [log](deep-preln-b0bbc2ae-8076-4bed-9d67-a768a423c29e.txt) |
| 2 | 3.33460 | 596.314 s | [log](deep-preln-2589ea6d-5b47-4303-aba0-4bf5672ab1d4.txt) |
| 3 | 3.33559 | 597.185 s | [log](deep-preln-697bc0fc-117a-43fa-90ac-880719c71f24.txt) |
| 4 | 3.33452 | 597.741 s | [log](deep-preln-5775b9aa-0722-4e64-a30a-05ab9f2f329b.txt) |
| **Mean** | **3.33480** | **596.86 s** | |

## How the steps and lr were chosen

- **Steps:** 2850 steps take about 600 s (`CALIBRATION_BUDGET` in `tracks.py`). Pre-LN runs at a steady
  209.8 ms/step on `deep`, against 162 ms/step for the L12 d768 `reference` shape on the same hardware.
- **lr:** a sweep of the Muon lr at 2850 steps (seed 0) found 0.025 best, in the middle of the range:

  | lr | 0.015 | 0.02 | **0.025** | 0.035 | 0.05 |
  | - | - | - | - | - | - |
  | val loss | 3.33989 | 3.33620 | **3.33398** | 3.33722 | 3.34933 |

- **Target:** 3.336 is just above the Pre-LN mean, so Pre-LN itself clears it at p < 0.01 and stands as record #1.
  The `reference` target follows the same rule: Pre-LN reaches 3.278 there and the target is 3.28.

The logs come from a checkout before Track 3 was added, so their embedded `tracks.py` has no `open` entry and
still says `target=None` for `deep`. `train_gpt.py`, `model.py` and `baselines/preln.py` are identical to the
current ones.
