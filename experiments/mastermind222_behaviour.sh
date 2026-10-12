#!/usr/bin/env bash
# Run the mastermind222_behaviour experiment (src/new_jax_shapley/configs/mastermind222_behaviour.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind222_behaviour.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m new_jax_shapley.main mastermind222_behaviour --compile-only "$@"
uv run python -m new_jax_shapley.main mastermind222_behaviour "$@"
