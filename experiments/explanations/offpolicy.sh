#!/usr/bin/env bash
# Pinned experiments/offpolicy/{off_policy,weighting,clipping}/dqn_{policy,value}_{mastermind,gwb}.py
# and offpolicy/run_policy_prediction.py: characteristics learnt from each seed's own DQN agent replay
# (paper Figures 3a and 7-9).
#
# Every seed trains one agent; upstream retrains the same seeded agent in every group, so all groups
# and kinds share it here. Each group then trains as one compiled sweep (seed s explains agent s):
#   mastermind222: exact (on-policy rollouts), wo_weight (replay, no IS), w_weight (replay, normalised IS)
#   gridworld:     exact, wo_weight, unnormalised (replay, IS without normalisation), normalised
#                  (replay, normalised IS; the clipping figure's "no clipping"), and clip_0.99,
#                  clip_0.995, clip_0.998 (normalised IS clipped to [1 - c, 1 + c])
# Usage: offpolicy.sh mastermind222|gridworld [--kind behaviour|prediction] [--num-seeds N]
#        [--output-root DIR] [--validate-only]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
domain=${1:-}
shift || true
dqn_gates "$@"
kinds=(behaviour prediction)
seeds=({1..20})
inspection=()
output_root=runs/explanations/offpolicy
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) kinds=("$2"); shift 2 ;;
        --num-seeds) seeds=($(seq 1 "$2")); shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --validate-only) inspection=("$1"); shift ;;
        *) echo "unsupported option $1" >&2; exit 2 ;;
    esac
done
case "$domain" in
    mastermind222)
        environment=(mastermind222) num_envs=16 timesteps=30000 exploration=0.25
        groups=(exact wo_weight w_weight)
        ;;
    gridworld)
        environment=(gridworld) num_envs=8 timesteps=15000 exploration=0.5
        groups=(exact wo_weight unnormalised normalised clip_0.99 clip_0.995 clip_0.998)
        ;;
    *) echo "usage: offpolicy.sh mastermind222|gridworld [options]" >&2; exit 2 ;;
esac
root="$output_root/$domain"
for seed in "${seeds[@]}"; do
    uv run python -m jax_shapley.train_dqn "${environment[@]}" \
        --seed "$seed" \
        --hidden-widths 64 64 \
        --train.num-envs "$num_envs" \
        --train.total-timesteps "$timesteps" \
        --train.buffer-size "$timesteps" \
        --train.learning-rate 0.00025 \
        --train.gamma 1 \
        --train.tau 0.01 \
        --train.target-network-frequency 1 \
        --train.batch-size 64 \
        --train.start-e 1 \
        --train.end-e 0.01 \
        --train.exploration-fraction "$exploration" \
        --train.learning-starts 10000 \
        --train.train-frequency 10 \
        --train.eval-episodes 0 \
        --train.adam.beta1 0.9 \
        --train.adam.beta2 0.999 \
        --train.adam.epsilon 0.00000001 \
        --output "$root/agents" \
        ${inspection[@]+"${inspection[@]}"}
done
seed_arguments=(--seeds "${seeds[@]}")
# The buffer is the whole replay: 80% trains, 20% validates; the exact buffer matches the train size.
train_size=$((timesteps * 4 / 5))
for kind in "${kinds[@]}"; do
    if [[ "$domain" == mastermind222 ]]; then
        epochs=2000
    elif [[ "$kind" == behaviour ]]; then
        epochs=50
    else
        epochs=100
    fi
    for group in "${groups[@]}"; do
        case "$group" in
            exact) data=() ;;
            wo_weight) data=(--data replay) ;;
            w_weight|normalised) data=(--data replay --off-policy) ;;
            unnormalised) data=(--data replay --off-policy --no-weighting) ;;
            clip_*) data=(--data replay --off-policy --clip-coef "${group#clip_}") ;;
        esac
        uv run python -m jax_shapley.train_explanations characteristic \
            "${seed_arguments[@]}" \
            --agent "$root/agents" \
            --kind "$kind" \
            --policy greedy \
            --hidden-widths 64 64 \
            ${data[@]+"${data[@]}"} \
            --train.epochs "$epochs" \
            --train.patience "$epochs" \
            --train.learning-rate 0.00025 \
            --train.batch-size 128 \
            --train.buffer-size "$timesteps" \
            --train.val-fraction 0.2 \
            --train.num-envs "$num_envs" \
            --train.exact-loss \
            --train.exact-buffer-size "$train_size" \
            --output "$root/$kind/$group" \
            ${inspection[@]+"${inspection[@]}"}
    done
done
