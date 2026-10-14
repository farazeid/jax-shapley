"""Tracking backends (wandb or trackio): an experiment's local scratch folder, and reading runs back.

Stages call wandb or trackio directly through their shared calls (`init`, `Run.log`,
`Run.log_artifact`, `Run.use_artifact`, `Artifact.download`, `Run.finish`):
`tracking = wandb if track == "wandb" else trackio`.

Sources of truth: for wandb, the cloud. Every experiment keeps its local wandb files
(runs, caches, downloaded artifacts) in one scratch folder, deleted when the experiment
ends, so nothing persists locally. For trackio, its local store (`TRACKIO_DIR`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import trackio
import wandb
from wandb.sdk.lib.service.service_connection import WandbApiFailedError
from wandb.sdk.wandb_run import Run as WandbRun

type Track = Literal["wandb", "trackio"]
type Scalars = dict[str, float]


class Logger(Protocol):
    """What live logging needs of a run: wandb's and trackio's `Run.log`."""

    def log(self, data: dict[str, float], /, step: int | None = None) -> None: ...


class TrackedRun(NamedTuple):
    """A tracked run read back: its logged config (stage settings, nested) and its logged rows."""

    config: dict[str, object]
    rows: list[Scalars]


# Attempts at a tracking-service call (checkpoint uploads, reading runs back), and the wait
# before the first retry (doubling after each one).
UPLOAD_ATTEMPTS = 5
UPLOAD_BACKOFF_SECONDS = 30.0
# What a busy or unreachable wandb service raises from its public API.
TRANSIENT = (wandb.errors.Error, WandbApiFailedError, OSError)

# wandb's local data, all redirected into the experiment's scratch folder.
WANDB_DIRECTORIES = ("WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR", "WANDB_ARTIFACT_DIR")
# Set while a session is open: nested sessions (a stage inside an experiment) reuse its folder.
SESSION = "JAX_SHAPLEY_SESSION"


def upload(run: WandbRun | trackio.Run, directory: Path, name: str, kind: str) -> None:
    """Push a stage's checkpoint folder as an artifact, retrying transient backend failures.

    A stage uploads its checkpoint only after training, so one busy or unreachable service
    would otherwise discard hours of work. Other errors, and the last attempt's, propagate.
    """
    for attempt in range(UPLOAD_ATTEMPTS):
        try:
            run.log_artifact(str(directory), name=name, type=kind).wait()
            return
        except TRANSIENT as error:
            if attempt == UPLOAD_ATTEMPTS - 1:
                raise
            delay = UPLOAD_BACKOFF_SECONDS * 2**attempt
            print(f"Artifact {name} upload failed ({error}); retrying in {delay:.0f} s.", flush=True)
            time.sleep(delay)


@contextmanager
def session(track: Track, entity: str | None) -> Iterator[Path]:
    """The local working folder for checkpoints and downloads, deleted at exit (also on error).

    For wandb every local file wandb writes goes here, `entity` is the default entity and
    several runs may be open at once. A session opened inside another one reuses its
    folder and leaves deleting it to the outer one, so stages within an experiment share
    it: an artifact pushed by one stage and pulled by the next is not downloaded again.
    """
    if SESSION in os.environ:
        yield Path(os.environ[SESSION])
        return
    root = Path(tempfile.mkdtemp(prefix="jax-shapley-"))
    names = (SESSION, *WANDB_DIRECTORIES, "WANDB_ENTITY", "WANDB_REINIT")
    saved = {name: os.environ.get(name) for name in names}
    try:
        os.environ[SESSION] = str(root)
        if track == "wandb":
            for name in WANDB_DIRECTORIES:
                folder = root / "wandb" / name.removeprefix("WANDB_").lower()
                folder.mkdir(parents=True)
                os.environ[name] = str(folder)
            os.environ["WANDB_REINIT"] = "create_new"
            if entity is not None:
                os.environ["WANDB_ENTITY"] = entity
        yield root
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(root, ignore_errors=True)


def runs(track: Track, *, project: str, entity: str | None, group: str) -> dict[str, TrackedRun]:
    """Every run of an experiment group by run name: its config and its logged numbers, each row with `_step`."""
    found: dict[str, TrackedRun] = {}
    if track == "wandb":
        path = f"{entity}/{project}" if entity else project
        for attempt in range(UPLOAD_ATTEMPTS):
            try:
                for run in wandb.Api().runs(path, filters={"group": group}):
                    rows = [
                        {key: float(value) for key, value in row.items() if isinstance(value, int | float)}
                        for row in run.scan_history()
                    ]
                    found[run.name] = TrackedRun(dict(run.config), rows)
                return found
            except TRANSIENT as error:
                if attempt == UPLOAD_ATTEMPTS - 1:
                    raise
                delay = UPLOAD_BACKOFF_SECONDS * 2**attempt
                print(f"Reading group {group} failed ({error}); retrying in {delay:.0f} s.", flush=True)
                found.clear()
                time.sleep(delay)
    from trackio.sqlite_storage import SQLiteStorage

    # trackio run names need not be unique within a project; run ids are.
    for record in SQLiteStorage.get_run_records(project):
        run_id, name = record["id"], record["name"]
        if run_id is None or name is None:
            continue
        config = SQLiteStorage.get_run_config(project, run_id=run_id) or {}
        if config.get("_Group") != group:
            continue
        rows = [
            {
                "_step" if key == "step" else key: float(value)
                for key, value in log.items()
                if isinstance(value, int | float)
            }
            for log in SQLiteStorage.get_logs(project, run_id=run_id)
        ]
        found[name] = TrackedRun(config, rows)
    return found


def groups(track: Track, *, project: str, entity: str | None, prefix: str) -> list[str]:
    """Experiment groups whose name starts with `prefix`, oldest first (names end in a timestamp)."""
    if track == "wandb":
        path = f"{entity}/{project}" if entity else project
        names: set[str] = set()
        for attempt in range(UPLOAD_ATTEMPTS):
            try:
                # Filtered by the service: listing every run of a large project overloads it.
                filters = {"group": {"$regex": f"^{prefix}"}}
                names = {run.group for run in wandb.Api().runs(path, filters=filters) if run.group}
                break
            except TRANSIENT as error:
                if attempt == UPLOAD_ATTEMPTS - 1:
                    raise
                delay = UPLOAD_BACKOFF_SECONDS * 2**attempt
                print(f"Listing groups {prefix}* failed ({error}); retrying in {delay:.0f} s.", flush=True)
                time.sleep(delay)
    else:
        from trackio.sqlite_storage import SQLiteStorage

        names = {
            str((SQLiteStorage.get_run_config(project, run_id=record["id"]) or {}).get("_Group"))
            for record in SQLiteStorage.get_run_records(project)
            if record["id"] is not None
        }
    return sorted(name for name in names if name.startswith(prefix))
