#!/usr/bin/env bash
# Run the mastermind333_sampled_behaviour experiment (src/jax_shapley/configs/mastermind333_sampled_behaviour.py): compile every stage, then train and track it.
# Usage: bash experiments/mastermind333_sampled_behaviour.sh [--track wandb|trackio] [overrides]
set -euo pipefail

uv run python -m jax_shapley.main mastermind333_sampled_behaviour --compile-only "$@"
uv run python -m jax_shapley.main mastermind333_sampled_behaviour "$@"
