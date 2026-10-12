#!/usr/bin/env bash
# Run the gridworld_behaviour experiment (src/new_jax_shapley/configs/gridworld_behaviour.py): compile every stage, then train and track it.
# Usage: bash experiments/gridworld_behaviour.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m new_jax_shapley.main gridworld_behaviour --compile-only "$@"
uv run python -m new_jax_shapley.main gridworld_behaviour "$@"
