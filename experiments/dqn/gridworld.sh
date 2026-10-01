#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dqn_gates "$@"
uv run python -m jax_shapley.train_dqn gridworld \
    --seed 1 \
    --checkpoint-mode run \
    --hidden-widths 64 64 \
    --train.num-envs 1 \
    --train.total-timesteps 50000 \
    --train.buffer-size 50000 \
    --train.learning-rate 0.00025 \
    --train.gamma 1 \
    --train.tau 0.01 \
    --train.target-network-frequency 1 \
    --train.batch-size 64 \
    --train.start-e 1 \
    --train.end-e 0.01 \
    --train.exploration-fraction 0.5 \
    --train.learning-starts 10000 \
    --train.train-frequency 10 \
    --train.eval-episodes 0 \
    --train.adam.beta1 0.9 \
    --train.adam.beta2 0.999 \
    --train.adam.epsilon 0.00000001 \
    "$@"
