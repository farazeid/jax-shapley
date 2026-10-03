#!/usr/bin/env bash
# Pinned experiments/fastsverl/agents/dqn_mastermind.py: the large agents for Mastermind-333
# (1M steps) and Mastermind-443/453/463 (10M steps), 2x128 networks, one agent (seed 1).
# Usage: mastermind.sh 333|443|453|463 [overrides]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
size=${1:-}
shift || true
case "$size" in
    333) timesteps=1000000 ;;
    443|453|463) timesteps=10000000 ;;
    *) echo "usage: mastermind.sh 333|443|453|463 [overrides]" >&2; exit 2 ;;
esac
dqn_gates "$@"
uv run python -m jax_shapley.train_dqn mastermind \
    --env.code-size "${size:0:1}" \
    --env.num-guesses "${size:1:1}" \
    --env.num-pegs "${size:2:1}" \
    --seed 1 \
    --hidden-widths 128 128 \
    --train.num-envs 16 \
    --train.total-timesteps "$timesteps" \
    --train.buffer-size "$timesteps" \
    --train.learning-rate 0.00025 \
    --train.gamma 1 \
    --train.tau 0.01 \
    --train.target-network-frequency 1 \
    --train.batch-size 64 \
    --train.start-e 1 \
    --train.end-e 0.01 \
    --train.exploration-fraction 0.25 \
    --train.learning-starts 10000 \
    --train.train-frequency 10 \
    --train.eval-episodes 0 \
    --train.adam.beta1 0.9 \
    --train.adam.beta2 0.999 \
    --train.adam.epsilon 0.00000001 \
    "$@"
