#!/usr/bin/env bash
# Run the gridworld_sampled_outcome experiment (src/jax_shapley/configs/gridworld_sampled_outcome.py): compile every stage, then train and track it.
# Usage: bash experiments/gridworld_sampled_outcome.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main gridworld_sampled_outcome --compile-only "$@"
uv run python -m jax_shapley.main gridworld_sampled_outcome "$@"
