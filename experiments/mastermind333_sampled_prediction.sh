#!/usr/bin/env bash
# Run the mastermind333_sampled_prediction experiment (src/jax_shapley/configs/mastermind333_sampled_prediction.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind333_sampled_prediction.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main mastermind333_sampled_prediction --compile-only "$@"
uv run python -m jax_shapley.main mastermind333_sampled_prediction "$@"
