"""DepthDash tracks. Frozen: a submission changes only its residual module (and declares its --steps and --lr).

Every shape has (nearly) the same backbone, 12 * L * d^2 ~= 85M parameters (attention 4d^2 + MLP 8d^2 per layer),
so the tracks differ only in aspect ratio d/L. This is the iso-backbone setting of the DepthBench paper.
`reference` and `extreme_deep` are exactly 84.93M. `deep` takes the round shape L28 d512 at 88.08M (+3.7%);
L27 d512 would match exactly.

Scoring is time-to-target, as in modded-nanogpt: the fastest training run whose final val loss reaches the
track's target. Each target is calibrated from Pre-LN trained for CALIBRATION_BUDGET seconds, so
Pre-LN is record #1 at about that time.
"""
from dataclasses import dataclass
from typing import Optional

VOCAB_SIZE = 50304
SEQ_LEN = 1024
BATCH_TOKENS = 8 * 64 * 1024  # 524,288 tokens per step
VAL_TOKENS = 20 * 524288
TRAIN_FILES = "data/fineweb10B/fineweb_train_*.bin"
VAL_FILES = "data/fineweb10B/fineweb_val_*.bin"
CALIBRATION_BUDGET = 600  # seconds on 8xH100: Pre-LN's loss after this long sets each track's target


@dataclass(frozen=True)
class Track:
    num_layers: int
    model_dim: int
    head_dim: int
    steps: Optional[int]      # Pre-LN's calibrated step count; the default for --steps
    lr: float                 # Pre-LN's calibrated Muon lr; the default for --lr
    target: Optional[float]   # val loss to reach; None until calibrated
    microbatch: int           # sequences per forward per GPU: memory only, the math is unchanged

    @property
    def aspect_ratio(self) -> float:
        return self.model_dim / self.num_layers

    @property
    def backbone_params(self) -> int:
        return 12 * self.num_layers * self.model_dim ** 2


TRACKS = {
    # Upstream track 3's baseline shape: Pre-LN reaches 3.28 in 3250 steps. A regression check, not scored.
    "reference":    Track(num_layers=12, model_dim=768, head_dim=128, steps=3250, lr=0.025, target=3.28, microbatch=64),
    # AR 18.3, cf. the paper's L=42 (AR 19.05).
    "deep":         Track(num_layers=28, model_dim=512, head_dim=64, steps=None, lr=0.025, target=None, microbatch=64),
    # AR 8, cf. the paper's L=70 (AR 9.14).
    "extreme_deep": Track(num_layers=48, model_dim=384, head_dim=64, steps=None, lr=0.025, target=None, microbatch=64),
    # Open: anything goes, as in Modded-NanoGPT's main track. The deep shape and target are the starting point.
    "open":         Track(num_layers=28, model_dim=512, head_dim=64, steps=None, lr=0.025, target=None, microbatch=64),
}
