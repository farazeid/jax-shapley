"""Typed host-only JSON, CSV, and non-pickle checkpoint boundaries."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from jax_shapley.algorithms.dqn import RunState, StepMetrics, epsilon_at
from jax_shapley.config import DQNRunConfig, EnvironmentConfig
from jax_shapley.networks import MLPConfig, QNetwork
from jax_shapley.replay import ReplayState, TransitionBatch

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]


@dataclass(frozen=True, slots=True)
class RunDescription[Config: EnvironmentConfig]:
    config: DQNRunConfig[Config]
    network: MLPConfig
    environment: str
    reference_commit: str = REFERENCE_COMMIT


@dataclass(frozen=True, slots=True)
class LoadedAgent:
    model: QNetwork
    target: QNetwork
    epsilon: float
    vector_step: int
    replay: ReplayState | None


@dataclass(frozen=True, slots=True)
class MetricRow:
    kind: Literal["episode", "update", "evaluation"]
    step: int
    epsilon: float | None = None
    loss: float | None = None
    episode_return: float | None = None
    episode_length: int | None = None


METRIC_COLUMNS = ("kind", "step", "epsilon", "loss", "episode_return", "episode_length")


def describe[Config: EnvironmentConfig](config: DQNRunConfig[Config]) -> str:
    return (
        json.dumps(
            asdict(RunDescription(config, config.network, type(config.env).__name__)), default=_json_default, indent=2
        )
        + "\n"
    )


def save_checkpoint[State, Config: EnvironmentConfig](
    folder: Path, state: RunState[State], config: DQNRunConfig[Config]
) -> None:
    """Atomically replace each file; loaders reject interrupted mixed-step pairs."""
    step = host_array(state.vector_step)
    version = np.asarray(1, dtype=np.int32)
    if config.checkpoint_mode != "none":
        epsilon = epsilon_at(config.train, jnp.maximum(state.vector_step - 1, 0) * config.train.num_envs)
        arrays: dict[str, HostArray] = {"version": version, "vector_step": step, "epsilon": host_array(epsilon)}
        for index, (source, target) in enumerate(zip(state.model.mlp.layers, state.target.mlp.layers, strict=True)):
            arrays[f"kernel_{index}"] = host_array(source.kernel[...])
            arrays[f"bias_{index}"] = host_array(source.bias[...])
            arrays[f"target_kernel_{index}"] = host_array(target.kernel[...])
            arrays[f"target_bias_{index}"] = host_array(target.bias[...])
        _save_arrays(folder / "model.npz", arrays)
    if config.checkpoint_mode in ("agent-replay", "run"):
        data = state.replay.data
        replay = {
            "version": version,
            "vector_step": step,
            "pointer": host_array(state.replay.pointer),
            "size": host_array(state.replay.size),
            "observations": host_array(data.observations),
            "next_observations": host_array(data.next_observations),
            "actions": host_array(data.actions),
            "rewards": host_array(data.rewards),
            "terminated": host_array(data.terminated),
            "log_probabilities": host_array(data.log_probabilities),
        }
        _save_arrays(folder / "replay.npz", replay)


def load_checkpoint(folder: Path, network: MLPConfig, *, load_replay: bool = True) -> LoadedAgent:
    model, target = QNetwork(network, rngs=nnx.Rngs(0)), QNetwork(network, rngs=nnx.Rngs(0))
    with np.load(folder / "model.npz", allow_pickle=False) as archive:
        if int(_read(archive, "version", (), np.dtype("int32"))) != 1:
            raise ValueError("unsupported model checkpoint version")
        step = int(_read(archive, "vector_step", (), np.dtype("int32")))
        epsilon = float(_read(archive, "epsilon", (), np.dtype("float32")))
        if step < 0 or not 0 <= epsilon <= 1:
            raise ValueError("invalid checkpoint step or epsilon")
        for index, (source, destination) in enumerate(zip(model.mlp.layers, target.mlp.layers, strict=True)):
            for prefix, layer in (("", source), ("target_", destination)):
                layer.kernel[...] = _read(
                    archive, f"{prefix}kernel_{index}", layer.kernel[...].shape, np.dtype("float32")
                )
                layer.bias[...] = _read(archive, f"{prefix}bias_{index}", layer.bias[...].shape, np.dtype("float32"))
    replay = _load_replay(folder, network.input_features, step) if load_replay else None
    return LoadedAgent(model, target, epsilon, step, replay)


def initialize_metrics(path: Path) -> None:
    with path.open("w", newline="") as output:
        csv.writer(output).writerow(METRIC_COLUMNS)


def record_metrics(path: Path, metrics: StepMetrics, num_envs: int) -> None:
    steps, active = host_array(metrics.global_step), host_array(metrics.active)
    epsilon, losses, updated = host_array(metrics.epsilon), host_array(metrics.loss), host_array(metrics.updated)
    completed, returns, lengths = (
        host_array(metrics.completed),
        host_array(metrics.episode_return),
        host_array(metrics.episode_length),
    )
    rows: list[MetricRow] = []
    for index in range(len(steps)):
        if not bool(active[index]):
            continue
        step = int(steps[index])
        rows.extend(
            MetricRow(
                "episode",
                step,
                epsilon=float(epsilon[index]),
                episode_return=float(returns[index, env]),
                episode_length=int(lengths[index, env]),
            )
            for env in range(num_envs)
            if bool(completed[index, env])
        )
        if bool(updated[index]) and step % 100 < num_envs:
            rows.append(MetricRow("update", step, epsilon=float(epsilon[index]), loss=float(losses[index])))
    write_metric_rows(path, rows)


def write_metric_rows(path: Path, rows: list[MetricRow]) -> None:
    with path.open("a", newline="") as output:
        writer = csv.writer(output)
        for row in rows:
            writer.writerow((row.kind, row.step, row.epsilon, row.loss, row.episode_return, row.episode_length))


def host_array(value: Array) -> HostArray:
    return cast(HostArray, np.asarray(jax.device_get(value)))  # Explicit JAX-to-NumPy host boundary.


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value {type(value)}")


def _save_arrays(path: Path, arrays: dict[str, HostArray]) -> None:
    with TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        np.savez_compressed(staged, allow_pickle=False, **arrays)
        staged.replace(path)


def _load_replay(folder: Path, features: int, step: int) -> ReplayState:
    with np.load(folder / "replay.npz", allow_pickle=False) as archive:
        if int(_read(archive, "version", (), np.dtype("int32"))) != 1:
            raise ValueError("unsupported replay checkpoint version")
        if int(_read(archive, "vector_step", (), np.dtype("int32"))) != step:
            raise ValueError("model/replay checkpoint steps disagree")
        obs = cast(HostArray, archive["observations"])  # Validate file-supplied dimensions before constructing state.
        if obs.ndim != 2 or obs.shape[1] != features or obs.shape[0] < 1:
            raise ValueError("invalid replay observation shape")
        capacity = obs.shape[0]
        pointer = _read(archive, "pointer", (), np.dtype("int32"))
        size = _read(archive, "size", (), np.dtype("int32"))
        if not 0 <= int(pointer) < capacity or not 0 <= int(size) <= capacity:
            raise ValueError("invalid replay pointer/size")
        data = TransitionBatch(
            _read(archive, "observations", (capacity, features), np.dtype("int32")),
            _read(archive, "actions", (capacity,), np.dtype("int32")),
            _read(archive, "rewards", (capacity,), np.dtype("float32")),
            _read(archive, "next_observations", (capacity, features), np.dtype("int32")),
            _read(archive, "terminated", (capacity,), np.dtype("bool")),
            _read(archive, "log_probabilities", (capacity,), np.dtype("float32")),
        )
        data.validate()
        return ReplayState(data, pointer, size)


def _read(archive: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...], dtype: np.dtype[np.generic]) -> Array:
    value = cast(HostArray, archive[name])  # Dynamic archive boundary, validated below.
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(f"invalid checkpoint field {name}: expected {shape} {dtype}")
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite checkpoint field {name}")
    return jnp.asarray(value)
