"""Every registered experiment config loads, and every run script supplies what its config leaves open.

Each `experiments/*.sh` runs with a stand-in `uv` that records its commands. Every recorded
`new_jax_shapley.main` command must parse into a complete, valid experiment config: sweep
scripts supply each of their config's MISSING fields (and only real fields).
"""

import dataclasses
import os
import shlex
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local argv, no shell interpolation.
from pathlib import Path

import pytest
import tyro

from new_jax_shapley.experiment import load_configs

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(path for path in (ROOT / "experiments").glob("*.sh") if path.name != "smoke.sh")


def test_every_config_has_a_run_script() -> None:
    assert {path.stem for path in SCRIPTS} == set(load_configs())


@pytest.mark.parametrize("script", SCRIPTS, ids=[path.stem for path in SCRIPTS])
def test_run_scripts_complete_their_configs(script: Path, tmp_path: Path) -> None:
    record = tmp_path / "commands"
    fake = tmp_path / "bin" / "uv"
    fake.parent.mkdir()
    fake.write_text(
        f'#!/usr/bin/env bash\nprintf "%q " "$@" >> {shlex.quote(str(record))}\necho >> {shlex.quote(str(record))}\n'
    )
    fake.chmod(0o755)
    environment = {**os.environ, "PATH": f"{fake.parent}{os.pathsep}{os.environ['PATH']}"}
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, str(script)], env=environment, check=True, cwd=ROOT)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local argv.
    commands = [shlex.split(line) for line in record.read_text().splitlines() if line.strip()]
    assert commands
    configs = load_configs()
    for command in commands:
        assert command[:4] == ["run", "python", "-m", "new_jax_shapley.main"], command
        config = tyro.extras.overridable_config_cli(configs, args=command[4:])
        assert config.name == script.stem
        assert not _missing(config), _missing(config)


def _missing(value: object, path: str = "") -> list[str]:
    """Dotted paths of every field still MISSING in a (nested) config."""
    if value is tyro.MISSING:
        return [path]
    if isinstance(value, tuple):
        return [found for index, item in enumerate(value) for found in _missing(item, f"{path}.{index}")]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            found
            for field in dataclasses.fields(value)
            for found in _missing(getattr(value, field.name), f"{path}.{field.name}")
        ]
    return []
