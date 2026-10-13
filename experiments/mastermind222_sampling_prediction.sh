#!/usr/bin/env bash
# Run the mastermind222_sampling_prediction experiment (src/jax_shapley/configs/mastermind222_sampling_prediction.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind222_sampling_prediction.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main mastermind222_sampling_prediction --compile-only "$@"
uv run python -m jax_shapley.main mastermind222_sampling_prediction "$@"
