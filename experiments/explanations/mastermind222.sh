#!/usr/bin/env bash
# Pinned experiments/fastsverl/{characteristics,shapleys}/dqn_{policy,value}_mastermind.py (2 2 2).
# Usage: mastermind222.sh characteristic --kind behaviour|prediction --agent RUN [overrides]
#        mastermind222.sh shapley --agent RUN --characteristic RUN [overrides]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
stage=${1:-}
shift || true
# Characteristic epochs (behaviour, prediction), Shapley epochs, patience.
explanation_stage "$stage" 50 80 50 10000 -- "$@"
