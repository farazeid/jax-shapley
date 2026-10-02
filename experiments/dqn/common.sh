#!/usr/bin/env bash
# Source from recipes; config values live in their CLI argument lists.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

dqn_gates() {
    # Inspection modes do not launch training jobs; honor the last toggle. Orchestrators that
    # already ran ./check.sh for this checkout set JAX_SHAPLEY_GATES_PASSED=1.
    if [[ "${JAX_SHAPLEY_GATES_PASSED:-}" == 1 ]]; then return 0; fi
    local validate=false compile=false argument
    for argument in "$@"; do
        case "$argument" in
            --validate-only) validate=true ;;
            --no-validate-only) validate=false ;;
            --compile-only) compile=true ;;
            --no-compile-only) compile=false ;;
            --help|-h) return 0 ;;
        esac
    done
    if [[ "$validate" == true || "$compile" == true ]]; then return 0; fi
    ./check.sh
}

# Read a selector without consuming arguments: Tyro still parses/validates it.
dqn_option() {
    local name=$1 value="" argument
    shift
    while [[ $# -gt 0 ]]; do
        argument=$1
        shift
        if [[ "$argument" == "$name" ]]; then
            if [[ $# -eq 0 ]]; then
                echo "Missing value for $name" >&2
                return 2
            fi
            value=$1
            shift
        elif [[ "$argument" == "$name="* ]]; then
            value=${argument#*=}
        fi
    done
    printf '%s' "$value"
}
