#!/usr/bin/env bash
# Run the mastermind333_prediction experiment (src/jax_shapley/configs/mastermind333_prediction.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind333_prediction.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main mastermind333_prediction --compile-only "$@"
uv run python -m jax_shapley.main mastermind333_prediction "$@"
