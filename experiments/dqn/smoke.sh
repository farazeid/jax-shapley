#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dqn_gates "$@"
uv run python -m jax_shapley.train_dqn mastermind222 \
    --seed 1 \
    --hidden-widths 4 \
    --chunk-steps 4 \
    --checkpoint-mode run \
    --train.num-envs 2 \
    --train.total-timesteps 32 \
    --train.buffer-size 7 \
    --train.batch-size 3 \
    --train.learning-starts 0 \
    --train.train-frequency 1 \
    --train.eval-episodes 2 \
    "$@"
