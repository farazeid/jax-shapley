#!/usr/bin/env bash
# Pinned experiments/fastsverl/{characteristics,shapleys}/dqn_{policy,value,performance}_gwb.py.
# Usage: gridworld.sh characteristic --kind behaviour|prediction --agent RUN [overrides]
#        gridworld.sh shapley --agent RUN --characteristic RUN [overrides]
#        gridworld.sh outcome --variant on_policy|off_policy --agent RUN --characteristic BEHAVIOUR_RUN [overrides]
#        gridworld.sh outcome-shapley --agent RUN --characteristic OUTCOME_RUN [overrides]
# The source's Gridworld outcome Shapley models explain the on-policy outcome characteristic.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
stage=${1:-}
shift || true
case "$stage" in
    # tau, outcome environments, on-policy timesteps, off-policy replay prefix, Shapley epochs.
    outcome|outcome-shapley) outcome_stage "$stage" 0.001 8 100000 50000 20 -- "$@" ;;
    # Characteristic epochs (behaviour, prediction), Shapley epochs, patience (empty: epochs).
    *) explanation_stage "$stage" 50 30 20 "" -- "$@" ;;
esac
