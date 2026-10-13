#!/usr/bin/env bash
# Run the mastermind222_outcome__sweep experiment (src/jax_shapley/configs/mastermind222_outcome__sweep.py)
# for each outcome variant (off- and on-policy) with its exact values conditioned on the exact
# and the learnt behaviour characteristic, compiling each first.
# Usage: bash experiments/mastermind222_outcome__sweep.sh [--variant off_policy|on_policy] [--with-exact-char True|False] [-- overrides]
set -euo pipefail

name=mastermind222_outcome__sweep
variants=(off_policy on_policy) choices=(True False) overrides=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) variants=("$2"); shift 2 ;;
        --with-exact-char) choices=("$2"); shift 2 ;;
        --) shift; overrides=("$@"); break ;;
        *) echo "unsupported option $1 (pass experiment overrides after --)" >&2; exit 2 ;;
    esac
done

for variant in "${variants[@]}"; do
    for exact in "${choices[@]}"; do
        swept=(--outcome.variant "$variant" --outcome.with-exact-char "$exact")
        uv run python -m jax_shapley.main "$name" "${swept[@]}" --compile-only ${overrides[@]+"${overrides[@]}"}
        uv run python -m jax_shapley.main "$name" "${swept[@]}" ${overrides[@]+"${overrides[@]}"}
    done
done
