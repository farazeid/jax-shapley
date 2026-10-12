#!/usr/bin/env bash
# Run the mastermind222_sampling_behaviour experiment (src/new_jax_shapley/configs/mastermind222_sampling_behaviour.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind222_sampling_behaviour.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m new_jax_shapley.main mastermind222_sampling_behaviour --compile-only "$@"
uv run python -m new_jax_shapley.main mastermind222_sampling_behaviour "$@"
