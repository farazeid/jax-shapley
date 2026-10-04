#!/usr/bin/env bash
# Pinned experiments/parallel/dqn_{policy,value,performance}_{mastermind,gwb}.py: explanation models
# trained alongside the agent at 1, 2, 10 and 50 explanation updates per agent update (paper Figures
# 3b and 10). Runs the plotted groups: w_exact (behaviour, prediction) and w_policy (outcome).
# Usage: continual.sh mastermind222|gridworld behaviour|prediction|outcome [--ratio R] [overrides]
#   20 seeds as one sweep per ratio; 100,000 steps over 16 environments; batch 64; agent and
#   explanation networks 2x64; exact buffers of 10,000 explore-policy observations, refreshed every
#   1,000 steps (5,000 for outcome). Each seed's agent trains identically at every ratio.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
domain=${1:-}
kind=${2:-}
shift 2 || true
dqn_gates "$@"
ratios=(1 2 10 50)
arguments=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ratio) ratios=("$2"); shift 2 ;;
        *) arguments+=("$1"); shift ;;
    esac
done
case "$domain" in
    mastermind222) exploration=0.25 ;;
    gridworld) exploration=0.5 ;;
    *) echo "usage: continual.sh mastermind222|gridworld behaviour|prediction|outcome [options]" >&2; exit 2 ;;
esac
case "$kind" in
    behaviour|prediction) exact_frequency=1000 ;;
    outcome) exact_frequency=5000 ;;
    *) echo "usage: continual.sh DOMAIN behaviour|prediction|outcome [options]" >&2; exit 2 ;;
esac
for ratio in "${ratios[@]}"; do
    uv run python -m jax_shapley.train_continual "$domain" \
        --seeds {1..20} \
        --hidden-widths 64 64 \
        --train.num-envs 16 \
        --train.total-timesteps 100000 \
        --train.buffer-size 100000 \
        --train.learning-rate 0.00025 \
        --train.gamma 1 \
        --train.tau 0.01 \
        --train.target-network-frequency 1 \
        --train.batch-size 64 \
        --train.start-e 1 \
        --train.end-e 0.01 \
        --train.exploration-fraction "$exploration" \
        --train.learning-starts 0 \
        --train.train-frequency 10 \
        --train.eval-episodes 0 \
        --explain.kind "$kind" \
        --explain.update-ratio "$ratio" \
        --explain.with-exact-char \
        --explain.outcome-with-exact-char \
        --explain.exact-frequency "$exact_frequency" \
        --explain.exact-buffer-size 10000 \
        --explain.hidden-widths 64 64 \
        --explain.learning-rate 0.00025 \
        --explain.tau 0.01 \
        --explain.gamma 1 \
        ${arguments[@]+"${arguments[@]}"}
done
