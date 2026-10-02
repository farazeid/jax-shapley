#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'TEXT'
Usage: bash scripts/explanations_port/run.sh [--seeds N] [--cases CASE...] [--skip-jax] [--skip-upstream]

Reproduce the explanation-port comparison from the repository root. The pinned
FastSVERL commit is cloned into a temporary directory, removed on exit, and run
in an isolated PyTorch environment. Runs execute sequentially so that the timings
are comparable. CASE is gridworld, mastermind222 or hypercube (default: all).
--skip-jax and --skip-upstream keep that side's existing CSVs; the upstream side is deterministic.
TEXT
}

seeds=5
skip_jax=false
skip_upstream=false
cases=(gridworld mastermind222 hypercube)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --seeds) seeds=$2; shift 2 ;;
        --skip-jax) skip_jax=true; shift ;;
        --skip-upstream) skip_upstream=true; shift ;;
        --cases) shift; cases=(); while [[ $# -gt 0 && "$1" != --* ]]; do cases+=("$1"); shift; done ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ ! -f pyproject.toml || ! -d scripts/explanations_port ]]; then
    echo "Run this command from the repository root: bash scripts/explanations_port/run.sh" >&2
    exit 2
fi

fastsverl_repository=https://github.com/djeb20/fastsverl.git
fastsverl_commit=25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b
fastsverl_source_root=$(mktemp -d /tmp/jax-shapley-fastsverl.XXXXXX)
jax_runs=$(mktemp -d /tmp/jax-shapley-explanation-runs.XXXXXX)
cleanup() {
    for folder in "$fastsverl_source_root" "$jax_runs"; do
        if [[ -d $folder ]]; then rm -rf -- "$folder"; fi
    done
}
trap cleanup EXIT

git clone --quiet "$fastsverl_repository" "$fastsverl_source_root"
git -C "$fastsverl_source_root" checkout --quiet --detach "$fastsverl_commit"

uv sync --all-groups
uv lock --check
./check.sh
export JAX_SHAPLEY_GATES_PASSED=1
if [[ "$skip_jax" == false ]]; then
    JAX_PLATFORMS=cpu uv run python scripts/explanations_port/run_jax.py \
        --root "$jax_runs" --seeds "$seeds" --cases "${cases[@]}"
fi

if [[ "$skip_upstream" == false ]]; then
    uv run --no-project --python 3.12 \
        --with "torch==2.6.0" --with "gymnasium==0.29.1" --with "numpy<2" \
        --with tensorboard --with tqdm --with pandas --with matplotlib --with scipy --with wandb \
        --with-editable "$fastsverl_source_root/fastsverl" \
        python scripts/explanations_port/run_upstream.py \
        --source-root "$fastsverl_source_root" --seeds "$seeds" --cases "${cases[@]}"
fi

uv run --group benchmark scripts/explanations_port/compare.py
echo "Explanation-port comparison complete."
