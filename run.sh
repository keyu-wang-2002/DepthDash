#!/bin/sh
# Usage: ./run.sh --track deep --residual baselines/hc.py --steps N --lr LR [--seed S] [--val-every N] [--microbatch M] [--checkpoint]
torchrun --standalone --nproc_per_node=8 train_gpt.py "$@"
