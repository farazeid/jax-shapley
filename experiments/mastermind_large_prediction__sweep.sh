#!/usr/bin/env bash
# Run the mastermind_large_prediction__sweep experiment (src/jax_shapley/configs/mastermind_large_prediction__sweep.py) for Mastermind-443, -453
# and -463 (4 pegs, 3 colours; 4, 5 and 6 guesses), compiling each first.
# Usage: bash experiments/mastermind_large_prediction__sweep.sh [--guesses 4|5|6] [-- overrides]
set -euo pipefail

name=mastermind_large_prediction__sweep
guesses=(4 5 6) overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --guesses) guesses=("$2"); shift 2 ;;
        --) shift; overrides=("$@"); break ;;
        *) echo "unsupported option $1 (pass experiment overrides after --)" >&2; exit 2 ;;
    esac
done

for count in "${guesses[@]}"; do
    swept=()
    for stage in algorithm characteristic shapley.0; do
        swept+=(--"$stage".env.num-guesses "$count")
    done
    uv run python -m jax_shapley.main "$name" "${swept[@]}" --compile-only ${overrides[@]+"${overrides[@]}"}
    uv run python -m jax_shapley.main "$name" "${swept[@]}" ${overrides[@]+"${overrides[@]}"}
done
