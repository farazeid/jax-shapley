#!/usr/bin/env bash
# Run the gridworld_sampled_behaviour experiment (src/jax_shapley/configs/gridworld_sampled_behaviour.py): compile every stage, then train and track it.
# Usage: bash experiments/gridworld_sampled_behaviour.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main gridworld_sampled_behaviour --compile-only "$@"
uv run python -m jax_shapley.main gridworld_sampled_behaviour "$@"
