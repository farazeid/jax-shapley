#!/usr/bin/env bash
# Run the gridworld_sampling_behaviour experiment (src/new_jax_shapley/configs/gridworld_sampling_behaviour.py): compile every stage, then train and track it.
# Usage: bash experiments/gridworld_sampling_behaviour.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m new_jax_shapley.main gridworld_sampling_behaviour --compile-only "$@"
uv run python -m new_jax_shapley.main gridworld_sampling_behaviour "$@"
