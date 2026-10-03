#!/usr/bin/env bash
# Source from explanation recipes; experiment values live in their CLI argument lists.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../dqn/common.sh"

# Run one stage of a single-agent explanation sweep (Gridworld, Mastermind-222).
# Usage: explanation_stage STAGE CHARACTERISTIC_EPOCHS_BEHAVIOUR CHARACTERISTIC_EPOCHS_PREDICTION
#        SHAPLEY_EPOCHS PATIENCE_OR_EMPTY -- RECIPE_ARGS...
explanation_stage() {
    local stage=$1 behaviour_epochs=$2 prediction_epochs=$3 shapley_epochs=$4 patience=$5
    shift 6
    dqn_gates "$@"
    local epochs exact
    local -a seeds patience_args
    # All 20 paper seeds train as one compiled sweep; --seed N selects a single ordinary run.
    seeds=(--seeds {1..20})
    patience_args=()
    if [[ -n "$(dqn_option --seed "$@")" ]]; then seeds=(); fi
    if [[ -n "$patience" ]]; then patience_args=(--train.patience "$patience"); fi
    case "$stage" in
        characteristic)
            case "$(dqn_option --kind "$@")" in
                behaviour) epochs=$behaviour_epochs ;;
                prediction) epochs=$prediction_epochs ;;
                *) echo "characteristic runs require --kind behaviour|prediction" >&2; return 2 ;;
            esac
            uv run python -m jax_shapley.train_explanations characteristic \
                ${seeds[@]+"${seeds[@]}"} \
                --policy greedy \
                --hidden-widths 64 64 \
                --train.epochs "$epochs" \
                ${patience_args[@]+"${patience_args[@]}"} \
                --train.learning-rate 0.00025 \
                --train.batch-size 128 \
                --train.buffer-size 10000 \
                --train.val-fraction 0.2 \
                --train.num-envs 16 \
                --train.exact-loss \
                "$@"
            ;;
        shapley)
            # Paper protocol: --characteristic names one run that every seed explains. Naming a
            # characteristic sweep instead pairs Shapley seed s with characteristic seed s.
            for exact in --with-exact-char --no-with-exact-char; do
                uv run python -m jax_shapley.train_explanations shapley \
                    ${seeds[@]+"${seeds[@]}"} \
                    "$exact" \
                    --buffers fresh \
                    --hidden-widths 64 64 \
                    --train.epochs "$shapley_epochs" \
                    ${patience_args[@]+"${patience_args[@]}"} \
                    --train.learning-rate 0.00025 \
                    --train.batch-size 128 \
                    --train.buffer-size 10000 \
                    --train.val-fraction 0.2 \
                    --train.num-envs 16 \
                    --train.exact-loss \
                    "$@"
            done
            ;;
        *)
            echo "usage: STAGE must be characteristic or shapley" >&2
            return 2
            ;;
    esac
}

# Run one outcome stage (Gridworld, Mastermind-222): dqn_performance_* characteristics and Shapley models.
# Usage: outcome_stage STAGE TAU NUM_ENVS TOTAL_TIMESTEPS TRAIN_SIZE SHAPLEY_EPOCHS -- RECIPE_ARGS...
outcome_stage() {
    local stage=$1 tau=$2 num_envs=$3 total=$4 train_size=$5 shapley_epochs=$6
    shift 7
    dqn_gates "$@"
    local exact
    local -a seeds
    seeds=(--seeds {1..20})
    if [[ -n "$(dqn_option --seed "$@")" ]]; then seeds=(); fi
    case "$stage" in
        outcome)
            case "$(dqn_option --variant "$@")" in
                on_policy|off_policy) ;;
                *) echo "outcome runs require --variant on_policy|off_policy" >&2; return 2 ;;
            esac
            # Both source groups: exact values conditioned on the exact (w_exact) or learnt (wo_exact)
            # behaviour characteristic. Off-policy: epochs = total * batch // train_size, validation
            # = 10 * num_envs, patience = epochs; on-policy: train_V_on_policy over num_envs.
            for exact in --with-exact-char --no-with-exact-char; do
                uv run python -m jax_shapley.train_explanations outcome \
                    ${seeds[@]+"${seeds[@]}"} \
                    "$exact" \
                    --tau "$tau" \
                    --gamma 1.0 \
                    --hidden-widths 64 64 \
                    --train.learning-rate 0.00025 \
                    --train.batch-size 64 \
                    --train.exact-loss \
                    --train.total-timesteps "$total" \
                    --train.num-envs "$num_envs" \
                    --train.train-frequency 1 \
                    --train.target-network-frequency 1 \
                    --train.train-size "$train_size" \
                    --train.validation-size $((10 * num_envs)) \
                    --train.epochs $((total * 64 / train_size)) \
                    --train.track-batch \
                    "$@"
            done
            ;;
        outcome-shapley)
            # --characteristic names an outcome run (or sweep). The source's three groups explain the
            # outcome characteristic trained with (w_policy) or without (wo_policy) the exact behaviour
            # characteristic against exact outcome values, and wo_policy against the learnt one (wo_perf).
            for exact in --with-exact-char --no-with-exact-char; do
                uv run python -m jax_shapley.train_explanations shapley \
                    ${seeds[@]+"${seeds[@]}"} \
                    "$exact" \
                    --buffers fresh \
                    --hidden-widths 64 64 \
                    --train.epochs "$shapley_epochs" \
                    --train.learning-rate 0.00025 \
                    --train.batch-size 128 \
                    --train.buffer-size 10000 \
                    --train.val-fraction 0.2 \
                    --train.num-envs 16 \
                    --train.exact-loss \
                    "$@"
            done
            ;;
        *)
            echo "usage: STAGE must be outcome or outcome-shapley" >&2
            return 2
            ;;
    esac
}
