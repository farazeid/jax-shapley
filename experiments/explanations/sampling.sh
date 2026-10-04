#!/usr/bin/env bash
# Pinned experiments/sampling/{shapleys/dqn_{policy,value}_*,characteristics/dqn_performance_*}.py:
# learnt (model) against sampled characteristics (paper Figures 3c and 11). 20 seeds; greedy explained
# policy; 16 environments; buffers 10,000 observations (8,000 train, 2,000 validation); batch 128.
#
# Usage: sampling.sh DOMAIN characteristic --kind behaviour|prediction --agent RUN --output-root DIR [overrides]
#        sampling.sh DOMAIN shapley --kind behaviour|prediction --agent RUN --model SWEEP --sampled SWEEP
#                    --output-root DIR [overrides]
#        sampling.sh DOMAIN outcome --agent RUN --model SWEEP --sampled SWEEP --output-root DIR [overrides]
#   DOMAIN: mastermind222 | mastermind333 | gridworld (outcome: mastermind222 | gridworld).
#   characteristic: a learnt characteristic sweep (DIR/model) and a sampled one (DIR/sampled), which
#     only collects the pool: its train buffer, also the exact buffer.
#   shapley: the source's groups. w_exact_model and wo_exact_model explain the learnt sweep (seed s
#     explains characteristic seed s) on fresh buffers; w_exact_sampling explains the sampled sweep on
#     its own buffers. Same-seed source processes train the same characteristic for both model groups.
#   outcome: on-policy outcome characteristics conditioned on the behaviour sweeps (the source's
#     outcome flow trains its behaviour characteristic with these settings: 50 epochs, 64x64).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
domain=${1:-}
stage=${2:-}
shift 2 || true
dqn_gates "$@"
kind=$(dqn_option --kind "$@")
output_root=$(dqn_option --output-root "$@")
model=$(dqn_option --model "$@")
sampled=$(dqn_option --sampled "$@")
# Drop the recipe's own options; everything else passes through to every command.
arguments=()
skip=false
for argument in "$@"; do
    if [[ "$skip" == true ]]; then skip=false; continue; fi
    case "$argument" in
        --output-root|--model|--sampled) skip=true; continue ;;
    esac
    arguments+=("$argument")
done
if [[ -z "$output_root" ]]; then echo "sampling.sh needs --output-root DIR" >&2; exit 2; fi
seeds=(--seeds {1..20})
if [[ -n "$(dqn_option --seed "$@")" ]]; then seeds=(); fi
case "$domain:$kind" in
    mastermind222:behaviour) width=64 characteristic_epochs=50 shapley_epochs=50 patience=10000 ;;
    mastermind222:prediction) width=64 characteristic_epochs=80 shapley_epochs=50 patience=10000 ;;
    mastermind333:behaviour) width=128 characteristic_epochs=1000 shapley_epochs=300 patience=10000 ;;
    mastermind333:prediction) width=128 characteristic_epochs=2000 shapley_epochs=500 patience=10000 ;;
    gridworld:behaviour) width=64 characteristic_epochs=50 shapley_epochs=20 patience= ;;
    gridworld:prediction) width=64 characteristic_epochs=30 shapley_epochs=20 patience= ;;
    mastermind222:|gridworld:) width=64 ;;
    *) echo "usage: sampling.sh mastermind222|mastermind333|gridworld STAGE --kind behaviour|prediction ..." >&2; exit 2 ;;
esac
common=(
    --hidden-widths "$width" "$width"
    --train.learning-rate 0.00025
    --train.batch-size 128
    --train.buffer-size 10000
    --train.val-fraction 0.2
    --train.num-envs 16
)
case "$stage" in
    characteristic)
        # Gridworld's source patience equals the epochs; Mastermind's is 10,000.
        uv run python -m jax_shapley.train_explanations characteristic ${seeds[@]+"${seeds[@]}"} \
            --policy greedy "${common[@]}" \
            --train.epochs "$characteristic_epochs" --train.patience "${patience:-$characteristic_epochs}" \
            --train.exact-loss --output "$output_root/model" "${arguments[@]}"
        uv run python -m jax_shapley.train_explanations characteristic ${seeds[@]+"${seeds[@]}"} \
            --policy greedy --estimator sampled "${common[@]}" \
            --train.no-exact-loss --output "$output_root/sampled" "${arguments[@]}"
        ;;
    shapley)
        if [[ -z "$model" || -z "$sampled" ]]; then echo "shapley needs --model SWEEP --sampled SWEEP" >&2; exit 2; fi
        # The Shapley stage reads its kind from the characteristic; drop the recipe's --kind selector.
        explained=()
        skip=false
        for argument in "${arguments[@]}"; do
            if [[ "$skip" == true ]]; then skip=false; continue; fi
            if [[ "$argument" == --kind ]]; then skip=true; continue; fi
            explained+=("$argument")
        done
        training=(--train.epochs "$shapley_epochs" --train.patience "${patience:-$shapley_epochs}" --train.exact-loss)
        for group in w_exact_model wo_exact_model w_exact_sampling; do
            case "$group" in
                w_exact_model) source_run=(--characteristic "$model" --buffers fresh --with-exact-char) ;;
                wo_exact_model) source_run=(--characteristic "$model" --buffers fresh --no-with-exact-char) ;;
                w_exact_sampling) source_run=(--characteristic "$sampled" --buffers characteristic --with-exact-char) ;;
            esac
            uv run python -m jax_shapley.train_explanations shapley ${seeds[@]+"${seeds[@]}"} \
                "${source_run[@]}" "${common[@]}" "${training[@]}" \
                --output "$output_root/$group" ${explained[@]+"${explained[@]}"}
        done
        ;;
    outcome)
        if [[ -z "$model" || -z "$sampled" ]]; then echo "outcome needs --model SWEEP --sampled SWEEP" >&2; exit 2; fi
        case "$domain" in
            mastermind222) tau=0.01 total=1000000 ;;
            gridworld) tau=0.001 total=100000 ;;
            *) echo "sampled outcome runs are for mastermind222|gridworld" >&2; exit 2 ;;
        esac
        for group in w_exact_model wo_exact_model w_exact_sampling; do
            case "$group" in
                w_exact_model) source_run=(--characteristic "$model" --with-exact-char) ;;
                wo_exact_model) source_run=(--characteristic "$model" --no-with-exact-char) ;;
                w_exact_sampling) source_run=(--characteristic "$sampled" --with-exact-char) ;;
            esac
            uv run python -m jax_shapley.train_explanations outcome ${seeds[@]+"${seeds[@]}"} \
                --variant on_policy "${source_run[@]}" \
                --tau "$tau" \
                --gamma 1.0 \
                --hidden-widths 64 64 \
                --train.learning-rate 0.00025 \
                --train.batch-size 64 \
                --train.exact-loss \
                --train.total-timesteps "$total" \
                --train.num-envs 16 \
                --train.train-frequency 1 \
                --train.target-network-frequency 1 \
                --output "$output_root/$group" "${arguments[@]}"
        done
        ;;
    *)
        echo "usage: sampling.sh DOMAIN characteristic|shapley|outcome [options]" >&2
        exit 2
        ;;
esac
