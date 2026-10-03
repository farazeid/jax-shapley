#!/usr/bin/env bash
# Pinned experiments/fastsverl/{characteristics,shapleys}/dqn_{policy,value,performance}_mastermind.py (2 2 2).
# Usage: mastermind222.sh characteristic --kind behaviour|prediction --agent RUN [overrides]
#        mastermind222.sh shapley --agent RUN --characteristic RUN [overrides]
#        mastermind222.sh outcome --variant on_policy|off_policy --agent RUN --characteristic BEHAVIOUR_RUN [overrides]
#        mastermind222.sh outcome-shapley --agent RUN --characteristic OUTCOME_RUN [overrides]
# The source's Mastermind outcome Shapley models explain the off-policy outcome characteristic.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
stage=${1:-}
shift || true
case "$stage" in
    # tau, outcome environments, on-policy timesteps, off-policy replay prefix, Shapley epochs.
    outcome|outcome-shapley) outcome_stage "$stage" 0.01 16 200000 100000 100 -- "$@" ;;
    # Characteristic epochs (behaviour, prediction), Shapley epochs, patience.
    *) explanation_stage "$stage" 50 80 50 10000 -- "$@" ;;
esac
