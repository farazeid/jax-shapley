#!/usr/bin/env bash
# Run the mastermind222_offpolicy_behaviour__sweep experiment (src/new_jax_shapley/configs/mastermind222_offpolicy_behaviour__sweep.py) for each of the paper's
# off-policy groups (exact wo_weight w_weight), compiling each first.
# Usage: bash experiments/mastermind222_offpolicy_behaviour__sweep.sh [--group GROUP] [-- overrides]
set -euo pipefail

name=mastermind222_offpolicy_behaviour__sweep
groups=(exact wo_weight w_weight)
overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) groups=("$2"); shift 2 ;;
        --) shift; overrides=("$@"); break ;;
        *) echo "unsupported option $1 (pass experiment overrides after --)" >&2; exit 2 ;;
    esac
done

for group in "${groups[@]}"; do
    case "$group" in
        exact) swept=(rollouts False True inf) ;;
        wo_weight) swept=(replay False True inf) ;;
        w_weight|normalised) swept=(replay True True inf) ;;
        unnormalised) swept=(replay True False inf) ;;
        clip_*) swept=(replay True True "${group#clip_}") ;;
        *) echo "unknown group $group" >&2; exit 2 ;;
    esac
    arguments=(
        --characteristic.data "${swept[0]}"
        --characteristic.off-policy "${swept[1]}"
        --characteristic.weighting "${swept[2]}"
        --characteristic.clip-coef "${swept[3]}"
    )
    uv run python -m new_jax_shapley.main "$name" "${arguments[@]}" --compile-only ${overrides[@]+"${overrides[@]}"}
    uv run python -m new_jax_shapley.main "$name" "${arguments[@]}" ${overrides[@]+"${overrides[@]}"}
done
