"""Project-specific pytest command-line options."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--parity-output",
        type=Path,
        default=None,
        help="Write exhaustive transition-parity measurements to this CSV path.",
    )
