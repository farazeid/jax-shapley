#!/usr/bin/env bash
# Run the mastermind222_sampled_outcome experiment (src/jax_shapley/configs/mastermind222_sampled_outcome.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind222_sampled_outcome.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main mastermind222_sampled_outcome --compile-only "$@"
uv run python -m jax_shapley.main mastermind222_sampled_outcome "$@"
