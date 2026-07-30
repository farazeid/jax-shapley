#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/envs_port/run.sh

Run the complete environment-port reproduction pipeline from the repository root.
The pinned FastSVERL commit is cloned into a temporary directory and removed on
exit.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

if [[ ! -f pyproject.toml || ! -d scripts/envs_port ]]; then
    echo "Run this command from the repository root:" >&2
    echo "  bash scripts/envs_port/run.sh" >&2
    exit 2
fi

fastsverl_repository=https://github.com/djeb20/fastsverl.git
fastsverl_commit=25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b
fastsverl_source_root=$(mktemp -d /tmp/jax-shapley-fastsverl.XXXXXX)

cleanup() {
    if [[ -d $fastsverl_source_root ]]; then
        rm -rf -- "$fastsverl_source_root"
    fi
}
trap cleanup EXIT

git clone --quiet "$fastsverl_repository" "$fastsverl_source_root"
git -C "$fastsverl_source_root" checkout --quiet --detach "$fastsverl_commit"

uv sync --all-groups --extra cuda
uv lock --check
uv run ruff check .

uv run scripts/envs_port/collect_fastsverl_transitions.py \
    --source-root "$fastsverl_source_root"

JAX_PLATFORMS=cpu uv run --extra cuda pytest \
    --parity-output docs/data/envs-port--parity--cpu.csv

JAX_PLATFORMS=cuda uv run --extra cuda pytest \
    tests/test_env_transition_parity.py \
    -m gpu \
    --parity-output docs/data/envs-port--parity--gpu.csv

uv run --group benchmark scripts/envs_port/benchmark_throughput.py \
    --backend original \
    --source-root "$fastsverl_source_root" \
    --output docs/data/envs-port--throughput--gymnasium-cpu.csv

JAX_PLATFORMS=cpu uv run --all-groups --extra cuda \
    scripts/envs_port/benchmark_throughput.py \
    --backend jax \
    --output docs/data/envs-port--throughput--jax-cpu.csv

JAX_PLATFORMS=cuda uv run --all-groups --extra cuda \
    scripts/envs_port/benchmark_throughput.py \
    --backend jax \
    --output docs/data/envs-port--throughput--jax-gpu.csv

uv run --group benchmark scripts/envs_port/summarise_throughput.py
uv run --group benchmark scripts/envs_port/plot_throughput.py

echo "Environment-port reproduction complete."
