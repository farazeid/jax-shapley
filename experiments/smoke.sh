#!/usr/bin/env bash
# Every stage of a registered experiment, shrunk to a few updates each, tracked like a real run.
#
#   bash experiments/smoke.sh                        # mastermind222_behaviour, trackio project "smoke"
#   bash experiments/smoke.sh NAME --track wandb     # any config; later arguments override earlier ones
#
# DQN: 320 environment steps over 16 envs, learning from step 160 every 10 vector steps (2 updates).
# Characteristic and each Shapley sweep: 2 seeds, 160-row buffers (128 train rows), batch 64,
# 2 epochs, so 2 updates per epoch. Artifacts are pushed and pulled between stages as usual.
set -euo pipefail

name="${1:-mastermind222_behaviour}"
shift || true

stage_overrides() {
  local prefix="$1"
  echo "--${prefix}.seeds 1 2 --${prefix}.train.epochs 2 --${prefix}.train.buffer-size 160 --${prefix}.train.batch-size 64"
}

# shellcheck disable=SC2046 # word splitting of the override lists is intended.
exec uv run python -m jax_shapley.main "$name" \
  --track trackio --project smoke \
  --algorithm.total-timesteps 320 --algorithm.learning-starts 160 --algorithm.buffer-size 320 --algorithm.batch-size 16 \
  $(stage_overrides characteristic) \
  $(stage_overrides shapley.0) \
  $(stage_overrides shapley.1) \
  "$@"
