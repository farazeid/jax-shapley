#!/usr/bin/env bash
# Seconds-long agent -> characteristic -> Shapley chain on Mastermind-222 for local checks.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dqn_gates "$@"
root=${SMOKE_ROOT:-runs/explanations/smoke-$(date +%s)}
uv run python -m jax_shapley.train_dqn mastermind222 \
    --seed 1 --hidden-widths 8 --chunk-steps 4 \
    --train.num-envs 2 --train.total-timesteps 64 --train.buffer-size 16 --train.batch-size 4 \
    --train.learning-starts 0 --train.train-frequency 1 \
    --output "$root/agent" "$@"
small=(--hidden-widths 8 --train.epochs 4 --train.buffer-size 40 --train.num-envs 4 --train.batch-size 8)
if [[ " $* " == *" --validate-only "* ]]; then
    agent="$root/agent/pending" characteristic="$root/characteristic/pending"
elif [[ " $* " == *" --compile-only "* ]]; then
    exit 0
else
    agent=$(find "$root/agent" -mindepth 1 -maxdepth 1 -type d)
fi
uv run python -m jax_shapley.train_explanations characteristic \
    --agent "$agent" --kind behaviour --output "$root/characteristic" "${small[@]}" "$@"
if [[ " $* " != *" --validate-only "* ]]; then
    characteristic=$(find "$root/characteristic" -mindepth 1 -maxdepth 1 -type d)
fi
uv run python -m jax_shapley.train_explanations shapley \
    --agent "$agent" --characteristic "$characteristic" --output "$root/shapley" "${small[@]}" "$@"
