#!/usr/bin/env bash
# Pinned experiments/hypercube/{hypercube_policy,hypercube_value}.py and run.py: each seed trains
# its own DQN agent, then a characteristic, then a Shapley model on the characteristic's buffers.
# Selectors: --kind behaviour|prediction, --seed N, --env.side-length N, --env.dimension N.
# --output-root DIR places runs under DIR (default runs/explanations/hypercube).
# Inspection: --validate-only validates every stage; --compile-only compiles the agent stage only,
# because later stages compile against artifacts that do not exist yet.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dqn_gates "$@"
kinds=(behaviour prediction)
seeds=({1..20})
selected_side="" selected_dimension="" inspection=() output_root=runs/explanations/hypercube
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) kinds=("$2"); shift 2 ;;
        --seed) seeds=("$2"); shift 2 ;;
        --env.side-length) selected_side=$2; shift 2 ;;
        --env.dimension) selected_dimension=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --validate-only|--compile-only) inspection=("$1"); shift ;;
        *) echo "unsupported option $1" >&2; exit 2 ;;
    esac
done

single_run() {
    local runs=("$1"/*/)
    if [[ ${#runs[@]} -ne 1 || ! -d "${runs[0]}" ]]; then
        echo "expected exactly one run in $1" >&2
        return 1
    fi
    printf '%s' "${runs[0]%/}"
}

# Side length, dimension, agent total timesteps (shared with experiments/dqn/hypercube.sh).
matched=0
while read -r side dimension timesteps; do
    if [[ -n "$selected_side" && "$side" != "$selected_side" ]]; then continue; fi
    if [[ -n "$selected_dimension" && "$dimension" != "$selected_dimension" ]]; then continue; fi
    matched=$((matched + 1))
    for kind in "${kinds[@]}"; do
        # hypercube_policy.py stops Shapley training at exact loss 0.001; hypercube_value.py at 0.01.
        if [[ "$kind" == behaviour ]]; then
            shapley_kind=behaviour shapley_condition=0.001
        else
            shapley_kind=prediction shapley_condition=0.01
        fi
        for seed in "${seeds[@]}"; do
            root="$output_root/$shapley_kind/${side}x${dimension}/seed-$seed"
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
                --output "$root/agent" \
                ${inspection[@]+"${inspection[@]}"}
            if [[ "${inspection[*]-}" == --compile-only ]]; then continue; fi
            if [[ "${inspection[*]-}" == --validate-only ]]; then
                agent="$root/agent/pending" characteristic="$root/characteristic/pending"
            else
                agent=$(single_run "$root/agent")
            fi
            common=(
                --seed "$seed"
                --hidden-widths 128 128
                --train.epochs 10000
                --train.learning-rate 0.00025
                --train.batch-size 128
                --train.buffer-size 10000
                --train.val-fraction 0.016
                --train.num-envs 16
                --train.exact-loss
                --train.track-batch
            )
            uv run python -m jax_shapley.train_explanations characteristic \
                --agent "$agent" \
                --kind "$shapley_kind" \
                --policy greedy \
                --train.exact-buffer-size 100000 \
                --train.exact-condition 0.01 \
                --output "$root/characteristic" \
                "${common[@]}" \
                ${inspection[@]+"${inspection[@]}"}
            if [[ "${inspection[*]-}" != --validate-only ]]; then characteristic=$(single_run "$root/characteristic"); fi
            uv run python -m jax_shapley.train_explanations shapley \
                --agent "$agent" \
                --characteristic "$characteristic" \
                --no-with-exact-char \
                --buffers characteristic \
                --train.exact-condition "$shapley_condition" \
                --output "$root/shapley" \
                "${common[@]}" \
                ${inspection[@]+"${inspection[@]}"}
        done
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
if [[ $matched -eq 0 ]]; then
    echo "no paper Hypercube matches the selected side length and dimension" >&2
    exit 2
fi
