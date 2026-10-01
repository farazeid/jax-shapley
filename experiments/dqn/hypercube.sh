#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for argument in "$@"; do
    if [[ "$argument" == --help || "$argument" == -h ]]; then
        uv run python -m jax_shapley.train_dqn hypercube --help
        exit 0
    fi
done
dqn_gates "$@"
selected_seed=$(dqn_option --seed "$@")
--checkpoint-mode run \
selected_side=$(dqn_option --env.side-length "$@")
selected_dimension=$(dqn_option --env.dimension "$@")
seeds=({1..20})
if [[ -n "$selected_seed" ]]; then seeds=("$selected_seed"); fi
matched=0
# Pinned experiments/hypercube/hypercube_policy.py and hypercube_value.py.
# Side length, dimension, total timesteps. Keep values in shell, not a Python preset.
while read -r side dimension timesteps; do
    if [[ -n "$selected_side" && "$side" != "$selected_side" ]]; then continue; fi
    if [[ -n "$selected_dimension" && "$dimension" != "$selected_dimension" ]]; then continue; fi
    matched=$((matched + 1))
    for seed in "${seeds[@]}"; do
        uv run python -m jax_shapley.train_dqn hypercube \
            --env.side-length "$side" \
            --env.dimension "$dimension" \
            --seed "$seed" \
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
            --train.end-e 0.05 \
            --train.exploration-fraction 0.5 \
            --train.learning-starts "$((timesteps / 50))" \
            --train.train-frequency 10 \
            --train.eval-episodes 0 \
            --train.adam.beta1 0.9 \
            --train.adam.beta2 0.999 \
            --train.adam.epsilon 0.00000001 \
            "$@"
    done
done <<'TABLE'
2 2 100000
3 2 50000
4 2 50000
5 2 50000
6 2 50000
2 3 50000
3 3 50000
4 3 50000
5 3 100000
6 3 100000
2 4 50000
3 4 50000
4 4 100000
5 4 100000
6 4 500000
2 5 50000
3 5 100000
4 5 500000
5 5 500000
6 5 1000000
TABLE
if [[ "$matched" -eq 0 ]]; then
    echo "No paper Hypercube matches the requested side length/dimension." >&2
    exit 2
fi
