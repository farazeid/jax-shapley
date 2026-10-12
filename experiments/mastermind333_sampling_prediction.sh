#!/usr/bin/env bash
# Run the mastermind333_sampling_prediction experiment (src/new_jax_shapley/configs/mastermind333_sampling_prediction.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind333_sampling_prediction.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m new_jax_shapley.main mastermind333_sampling_prediction --compile-only "$@"
uv run python -m new_jax_shapley.main mastermind333_sampling_prediction "$@"
