#!/usr/bin/env bash
# Run the mastermind222_outcome__sweep experiment (src/new_jax_shapley/configs/mastermind222_outcome__sweep.py)
# with the outcome's exact values conditioned on the exact and then the learnt behaviour
# characteristic, compiling each first.
# Usage: bash experiments/mastermind222_outcome__sweep.sh [--with-exact-char True|False] [-- overrides]
set -euo pipefail

name=mastermind222_outcome__sweep
choices=(True False) overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-exact-char) choices=("$2"); shift 2 ;;
        --) shift; overrides=("$@"); break ;;
        *) echo "unsupported option $1 (pass experiment overrides after --)" >&2; exit 2 ;;
    esac
done

for exact in "${choices[@]}"; do
    uv run python -m new_jax_shapley.main "$name" --outcome.with-exact-char "$exact" --compile-only ${overrides[@]+"${overrides[@]}"}
    uv run python -m new_jax_shapley.main "$name" --outcome.with-exact-char "$exact" ${overrides[@]+"${overrides[@]}"}
done
