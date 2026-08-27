"""Project-specific pytest command-line options."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--parity-output",
        type=Path,
        default=None,
        help="Write exhaustive transition-parity measurements to this CSV path.",
    )


def pytest_cmdline_main(config: pytest.Config) -> pytest.ExitCode | None:
    requested_platforms = {platform.strip() for platform in os.environ.get("JAX_PLATFORMS", "").split(",")}
    if not requested_platforms.intersection({"cuda", "gpu"}):
        return None

    import jax

    jax_logger = logging.getLogger("jax._src.xla_bridge")
    logger_was_disabled = jax_logger.disabled
    jax_logger.disabled = True
    try:
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
    finally:
        jax_logger.disabled = logger_was_disabled
    if gpu_devices:
        return None

    terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
    message = "SKIPPED GPU test run: this command requires an available JAX GPU backend."
    if terminal_reporter is None:
        print(message, flush=True)
    else:
        terminal_reporter.write_line(message, yellow=True)
    return pytest.ExitCode.OK
