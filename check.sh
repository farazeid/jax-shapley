#!/usr/bin/env bash

set -euo pipefail
uv run ty check
uv run ruff check .
uv run ruff format --check .
uv run pytest
