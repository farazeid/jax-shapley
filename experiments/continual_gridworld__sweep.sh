#!/usr/bin/env bash
# Run the continual_gridworld__sweep experiment (src/jax_shapley/configs/continual_gridworld__sweep.py) for each explanation
# kind and update ratio (1, 2, 10, 50), compiling each first.
# Usage: bash experiments/continual_gridworld__sweep.sh [--kind behaviour|prediction|outcome] [--ratio N] [-- overrides]
set -euo pipefail

name=continual_gridworld__sweep
kinds=(behaviour prediction outcome) ratios=(1 2 10 50) overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kind) kinds=("$2"); shift 2 ;;
        --ratio) ratios=("$2"); shift 2 ;;
        --) shift; overrides=("$@"); break ;;
        *) echo "unsupported option $1 (pass experiment overrides after --)" >&2; exit 2 ;;
    esac
done

for kind in "${kinds[@]}"; do
    frequency=1000
    if [[ "$kind" == outcome ]]; then frequency=5000; fi
    for ratio in "${ratios[@]}"; do
        swept=(--explain.kind "$kind" --explain.update-ratio "$ratio" --explain.exact-frequency "$frequency")
        uv run python -m jax_shapley.continual "$name" "${swept[@]}" --compile-only ${overrides[@]+"${overrides[@]}"}
        uv run python -m jax_shapley.continual "$name" "${swept[@]}" ${overrides[@]+"${overrides[@]}"}
    done
done
