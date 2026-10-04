#!/usr/bin/env bash
# Pinned experiments/fastsverl/{characteristics,shapleys}/dqn_{policy,value}_mastermind.py: the
# Mastermind-333 accuracy runs (paper Figure 4) and the large Mastermind-443/453/463 runs (Tables 1
# and 3). Networks are 2x128; buffers 10,000 observations (8,000 train, 2,000 validation) from 16
# environments; learning rate 2.5e-4; batch 128; greedy explained policy.
#
# Usage: mastermind.sh SIZE characteristic --kind behaviour|prediction --agent RUN [overrides]
#        mastermind.sh SIZE shapley --agent RUN --characteristic RUN [overrides]
#   333: 20 seeds with exact losses. Characteristics train 1,000 (behaviour) or 2,000 (prediction)
#        epochs; Shapley models 300 or 500 epochs, with and without the exact characteristic.
#   443|453|463: 10 seeds, no exact values (infeasible); 50,000 epochs with patience 5,000.
#   Pass --kind to the shapley stage too: its epochs depend on the explained kind.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
size=${1:-}
stage=${2:-}
shift 2 || true
dqn_gates "$@"
kind=$(dqn_option --kind "$@")
case "$size" in
    333)
        seeds=(--seeds {1..20})
        exact=(--train.exact-loss)
        patience=10000
        case "$stage:$kind" in
            characteristic:behaviour) epochs=1000 ;;
            characteristic:prediction) epochs=2000 ;;
            shapley:behaviour) epochs=300 ;;
            shapley:prediction) epochs=500 ;;
            *) echo "mastermind.sh 333 needs a stage and --kind behaviour|prediction" >&2; exit 2 ;;
        esac
        ;;
    443|453|463)
        seeds=(--seeds {1..10})
        exact=(--train.no-exact-loss)
        epochs=50000
        patience=5000
        ;;
    *)
        echo "usage: mastermind.sh 333|443|453|463 characteristic|shapley [overrides]" >&2
        exit 2
        ;;
esac
if [[ -n "$(dqn_option --seed "$@")" ]]; then seeds=(); fi
common=(
    --hidden-widths 128 128
    --train.epochs "$epochs"
    --train.patience "$patience"
    --train.learning-rate 0.00025
    --train.batch-size 128
    --train.buffer-size 10000
    --train.val-fraction 0.2
    --train.num-envs 16
    "${exact[@]}"
)
case "$stage" in
    characteristic)
        uv run python -m jax_shapley.train_explanations characteristic \
            ${seeds[@]+"${seeds[@]}"} --policy greedy "${common[@]}" "$@"
        ;;
    shapley)
        # The Shapley stage reads its kind from the characteristic; drop the recipe's --kind selector.
        arguments=()
        skip=false
        for argument in "$@"; do
            if [[ "$skip" == true ]]; then skip=false; continue; fi
            if [[ "$argument" == --kind ]]; then skip=true; continue; fi
            if [[ "$argument" == --kind=* ]]; then continue; fi
            arguments+=("$argument")
        done
        flags=(--with-exact-char --no-with-exact-char)
        if [[ "$size" != 333 ]]; then flags=(--no-with-exact-char); fi
        for flag in "${flags[@]}"; do
            uv run python -m jax_shapley.train_explanations shapley \
                ${seeds[@]+"${seeds[@]}"} "$flag" --buffers fresh "${common[@]}" ${arguments[@]+"${arguments[@]}"}
        done
        ;;
    *)
        echo "usage: mastermind.sh SIZE characteristic|shapley [overrides]" >&2
        exit 2
        ;;
esac
