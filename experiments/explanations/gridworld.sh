#!/usr/bin/env bash
# Pinned experiments/fastsverl/{characteristics,shapleys}/dqn_{policy,value}_gwb.py.
# Usage: gridworld.sh characteristic --kind behaviour|prediction --agent RUN [overrides]
#        gridworld.sh shapley --agent RUN --characteristic RUN [overrides]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
stage=${1:-}
shift || true
# Characteristic epochs (behaviour, prediction), Shapley epochs, patience (empty: epochs).
explanation_stage "$stage" 50 30 20 "" -- "$@"
