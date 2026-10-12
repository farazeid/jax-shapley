#!/usr/bin/env bash
# Run the gridworld_prediction experiment (src/new_jax_shapley/configs/gridworld_prediction.py): compile every stage, then train and track it.
# Usage: bash experiments/gridworld_prediction.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m new_jax_shapley.main gridworld_prediction --compile-only "$@"
uv run python -m new_jax_shapley.main gridworld_prediction "$@"
