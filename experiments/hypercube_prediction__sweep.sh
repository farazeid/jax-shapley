#!/usr/bin/env bash
# Run the hypercube_prediction__sweep experiment (src/new_jax_shapley/configs/hypercube_prediction__sweep.py) for
# each of the paper's 20 cubes (side length x dimension, agent timesteps), compiling each first.
# Usage: bash experiments/hypercube_prediction__sweep.sh [--side-length N] [--dimension N] [-- overrides]
set -euo pipefail

name=hypercube_prediction__sweep
selected_side="" selected_dimension="" overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --side-length) selected_side=$2; shift 2 ;;
        --dimension) selected_dimension=$2; shift 2 ;;
        --) shift; overrides=("$@"); break ;;
        *) echo "unsupported option $1 (pass experiment overrides after --)" >&2; exit 2 ;;
    esac
done

matched=0
while read -r side dimension timesteps; do
    if [[ -n "$selected_side" && "$side" != "$selected_side" ]]; then continue; fi
    if [[ -n "$selected_dimension" && "$dimension" != "$selected_dimension" ]]; then continue; fi
    matched=$((matched + 1))
    swept=(
        --algorithm.total-timesteps "$timesteps"
        --algorithm.buffer-size "$timesteps"
        --algorithm.learning-starts "$((timesteps / 50))"
    )
    for stage in algorithm characteristic shapley.0; do
        swept+=(--"$stage".env.side-length "$side" --"$stage".env.dimension "$dimension")
    done
    uv run python -m new_jax_shapley.main "$name" "${swept[@]}" --compile-only ${overrides[@]+"${overrides[@]}"}
    uv run python -m new_jax_shapley.main "$name" "${swept[@]}" ${overrides[@]+"${overrides[@]}"}
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
