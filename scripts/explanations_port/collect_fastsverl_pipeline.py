# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy<2", "gymnasium==0.29.1"]
# ///
"""Capture upstream buffer collection and train_models behaviour for pipeline parity.

Executes the unchanged `fill_buffer`, `get_data` and `train_models` functions and
the unchanged model classes at the pinned commit. Every random draw the loop
makes (buffer rows, coalitions) is recorded, so JAX can replay them exactly:

- buffers: 16 vector Hypercube(3, 5) environments with a hand-built,
  tie-free greedy agent that reaches the goal every 10 steps, exercising auto-reset
- loop: a behaviour characteristic and a Shapley model with small patience, and a
  prediction characteristic stopped by the exact-loss condition under track_batch

Buffer row draws are recorded by re-issuing upstream `Buffer.sample`'s own
`np.random.choice` call, which consumes the same global stream.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed read-only git operations.
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import numpy as np

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
SOURCES = ("characteristics.py", "shapley.py", "utils.py", "models.py", "dqn.py", "training.py", "envs/hypercube.py")
SEED = 1729
LEARNING_RATE = 2.5e-4
BUFFER_ENVS, BUFFER_STEPS = 16, 40
# Loop cases: Mastermind-222 observations (8 features, 4 actions), 32 train and 8 validation rows.
TRAIN_ROWS, VALIDATION_ROWS, BATCH_SIZE = 32, 8, 8


class Tensor(Protocol):
    grad: Tensor | None
    data: Tensor

    @property
    def shape(self) -> tuple[int, ...]: ...
    def detach(self) -> Tensor: ...
    def cpu(self) -> Tensor: ...
    def numpy(self) -> FloatArray: ...
    def copy_(self, other: Tensor) -> Tensor: ...
    def __getitem__(self, index: object) -> Tensor: ...


class SaveArrays(Protocol):
    def __call__(self, file: Path, /, **arrays: HostArray) -> None: ...


class Builder(Protocol):
    def SimpleNN(self, name: str, output_shape: tuple[int, ...], **architecture: object) -> Model: ...  # ruff: ignore[invalid-function-name] -- source constructor.


class Torch(Protocol):
    __version__: str

    def manual_seed(self, seed: int) -> object: ...
    def set_num_threads(self, count: int) -> None: ...
    def from_numpy(self, values: HostArray) -> Tensor: ...
    def device(self, name: str) -> object: ...
    def no_grad(self) -> object: ...


class Model(Protocol):
    def parameters(self) -> Iterator[Tensor]: ...


class Agent(Protocol):
    model: Model
    epsilon: float
    args: object


class Buffer(Protocol):
    size: int
    buffer: dict[str, Tensor]
    sample: Callable[..., list[Tensor]]

    def add(self, **values: object) -> None: ...


class Sampler(Protocol):
    sample: Callable[..., Tensor]
    sample_rand: Callable[..., Tensor]


class Learned(Protocol):
    model: Model
    sampler: Sampler


class Shapley(Protocol):
    model: Model
    char: Learned


class Writer:
    """TensorBoard stand-in that keeps every scalar train_models logs."""

    def __init__(self) -> None:
        self.scalars: dict[str, list[tuple[int, float]]] = defaultdict(list)

    def add_scalar(self, tag: str, value: object, step: int) -> None:
        self.scalars[tag].append((int(step), float(cast(float, value))))

    def add_text(self, *_: object) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--explanations", type=Path, default=Path("tests/fixtures/fastsverl_explanations.npz"))
    args = parser.parse_args()
    root, output, explanations = Path(args.source_root), Path(args.output), Path(args.explanations)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required")
    commit = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
    if commit != REFERENCE_COMMIT:
        raise ValueError("unexpected reference commit")
    hashes: dict[str, str] = {}
    for name in SOURCES:
        relative = f"fastsverl/fastsverl/{name}"
        committed = subprocess.check_output([git, "-C", str(root), "show", f"HEAD:{relative}"])  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
        if committed != (root / relative).read_bytes():
            raise ValueError(f"modified reference {name}")
        hashes[name] = hashlib.sha256(committed).hexdigest()
    explanations_sha256 = hashlib.sha256(explanations.read_bytes()).hexdigest()
    if explanations_sha256 != json.loads(explanations.with_suffix(".json").read_text())["fixture_sha256"]:
        raise ValueError("explanation fixture does not match its metadata")
    with np.load(explanations, allow_pickle=False) as fixture:
        reference = {key: fixture[key] for key in fixture.files}
    arrays, version, stops = capture(root, reference)
    # NumPy < 2 (required by gymnasium 0.29) would store an allow_pickle keyword as an array.
    cast(SaveArrays, np.savez_compressed)(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "fastsverl_commit": commit,
                "source_sha256": hashes,
                "explanations_sha256": explanations_sha256,
                "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "torch_version": version,
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "buffer": {"environments": BUFFER_ENVS, "steps": BUFFER_STEPS, "hypercube": [3, 5]},
                "loop": {"train_rows": TRAIN_ROWS, "validation_rows": VALIDATION_ROWS, "batch_size": BATCH_SIZE},
                "exact_condition": stops,
                "collector": "unchanged fill_buffer/get_data/train_models and model ASTs, CPU, recorded draws",
            },
            indent=2,
        )
        + "\n"
    )


def capture(root: Path, reference: dict[str, HostArray]) -> tuple[dict[str, HostArray], str, float]:
    torch = cast(Torch, importlib.import_module("torch"))  # Optional reference-only boundary.
    torch.set_num_threads(1)
    namespace = _namespace(root, torch)
    arrays: dict[str, HostArray] = {}
    arrays.update(capture_buffers(namespace, torch))
    loop, threshold = capture_loop(namespace, torch, reference)
    arrays.update(loop)
    return arrays, torch.__version__, threshold


def capture_buffers(namespace: dict[str, object], torch: Torch) -> dict[str, HostArray]:
    """Upstream fill_buffer over vector Hypercube(3, 5) with a goal-reaching greedy agent."""
    gymnasium = importlib.import_module("gymnasium")
    hypercube = cast(Callable[..., object], namespace["Hypercube"])

    statistics = cast(Callable[[object], object], gymnasium.wrappers.RecordEpisodeStatistics)
    vector = cast(Callable[[list[Callable[[], object]]], object], gymnasium.vector.SyncVectorEnv)

    def make() -> object:
        return statistics(hypercube(3, 5))

    envs = vector([make] * BUFFER_ENVS)
    agent = _agent(namespace, torch, envs, features=5, actions=10, width=64)
    weights = goal_seeking_weights(features=5, width=64)
    for parameter, value in zip(agent.model.parameters(), weights, strict=True):
        parameter.data.copy_(torch.from_numpy(value))
    agent.args = SimpleNamespace(verbose=0)
    buffer_cls = cast(Callable[..., Buffer], namespace["Buffer"])
    rows = BUFFER_ENVS * BUFFER_STEPS
    buffer = buffer_cls(rows, envs, "obs")
    fill = cast(Callable[..., tuple[Buffer, int]], namespace["fill_buffer"])
    fill(agent, envs, buffer, rows, "greedy", "obs")
    arrays: dict[str, HostArray] = {f"buffer_agent_{index}": value for index, value in enumerate(weights)}
    arrays["buffer_observations"] = buffer.buffer["obs"][: buffer.size].cpu().numpy().astype(np.int32)
    return arrays


def goal_seeking_weights(*, features: int, width: int) -> list[FloatArray]:
    """Greedy action: +1 along the lowest coordinate (lowest index on ties); -1 moves are never greedy.

    Upstream action 2i adds one along axis i, action 2i + 1 subtracts one. Both hidden
    ReLU layers pass the non-negative observation through unchanged.
    """
    first = np.zeros((width, features), np.float32)
    first[:features, :features] = np.eye(features)
    second = np.zeros((width, width), np.float32)
    second[:features, :features] = np.eye(features)
    output = np.zeros((2 * features, width), np.float32)
    bias = np.full(2 * features, -1000.0, np.float32)
    for axis in range(features):
        output[2 * axis, axis] = -1.0
        bias[2 * axis] = -0.001 * axis
    return [first, np.zeros(width, np.float32), second, np.zeros(width, np.float32), output, bias]


def capture_loop(
    namespace: dict[str, object], torch: Torch, reference: dict[str, HostArray]
) -> tuple[dict[str, HostArray], float]:
    """train_models for a behaviour characteristic, its Shapley model, and an exact-stopped prediction characteristic."""
    env = SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(8,)), single_action_space=SimpleNamespace(n=4, shape=())
    )
    observations = np.resize(reference["mastermind222_exact_buffer"], (TRAIN_ROWS + VALIDATION_ROWS, 8))
    arrays: dict[str, HostArray] = {
        "loop_train": observations[:TRAIN_ROWS].astype(np.int32),
        "loop_validation": observations[TRAIN_ROWS:].astype(np.int32),
    }
    for index in range(6):
        arrays[f"loop_agent_{index}"] = reference[f"mastermind222_agent_{index}"]
    agent = _agent(namespace, torch, env, features=8, actions=4, width=64)
    for parameter, index in zip(agent.model.parameters(), range(6), strict=True):
        parameter.data.copy_(torch.from_numpy(reference[f"mastermind222_agent_{index}"]))
    agent.epsilon = 0.2

    behaviour, records = _train_characteristic(
        namespace, torch, env, agent, observations, "PolicyCharacteristicModel", {}
    )
    arrays.update(_records("loop_behaviour", records))
    shapley_records = _train_shapley(namespace, torch, env, behaviour, observations)
    arrays.update(_records("loop_shapley", shapley_records))

    # Prediction characteristic: pick a threshold that stops mid-run, then rerun from the same seed.
    stopping: dict[str, object] = {"track_batch": True, "patience": 12, "epochs": 12}
    _, probe = _train_characteristic(namespace, torch, env, agent, observations, "ValueCharacteristicModel", stopping)
    scalars = cast(dict[str, list[tuple[int, float]]], probe["scalars"])
    exact = [value for _, value in scalars["model/ValueCharacteristicModel_exact_loss"]]
    stop = next(epoch for epoch in range(3, len(exact)) if exact[epoch] < min(exact[:epoch]))
    threshold = (exact[stop] + min(exact[:stop])) / 2
    _, records = _train_characteristic(
        namespace,
        torch,
        env,
        agent,
        observations,
        "ValueCharacteristicModel",
        {**stopping, "exact_condition": threshold},
    )
    arrays.update(_records("loop_prediction", records))
    return arrays, threshold


def _train_characteristic(
    namespace: dict[str, object],
    torch: Torch,
    env: SimpleNamespace,
    agent: Agent,
    observations: HostArray,
    name: str,
    overrides: dict[str, object],
) -> tuple[Learned, dict[str, object]]:
    train, validation = _buffers(namespace, env, observations)
    softmax = name == "PolicyCharacteristicModel"
    args = SimpleNamespace(
        model_arch=_architecture(8, 64, softmax=softmax),
        learning_rate=LEARNING_RATE,
        policy="greedy",
        verbose=0,
        epochs=8,
        patience=2,
        batch_size=BATCH_SIZE,
        train_buffer_size=TRAIN_ROWS,
        val_buffer_size=VALIDATION_ROWS,
        exact_loss=True,
        save_val=True,
        save_model=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    torch.manual_seed(SEED)
    np.random.seed(SEED)  # Upstream samples buffer rows from the legacy global stream.
    cls = cast(Callable[..., Learned], namespace[name])
    model = cls(agent, env, args, train, validation, train)
    records = _record(model, train, validation, "sample_rand")
    _run(namespace, model, args, records)
    return model, records


def _train_shapley(
    namespace: dict[str, object], torch: Torch, env: SimpleNamespace, char: Learned, observations: HostArray
) -> dict[str, object]:
    train, validation = _buffers(namespace, env, observations)
    args = SimpleNamespace(
        model_arch=_architecture(8, 64, softmax=False),
        learning_rate=LEARNING_RATE,
        with_exact_char=True,
        verbose=0,
        epochs=8,
        patience=2,
        batch_size=BATCH_SIZE,
        train_buffer_size=TRAIN_ROWS,
        val_buffer_size=VALIDATION_ROWS,
        exact_loss=True,
        save_val=True,
        save_model=False,
    )
    torch.manual_seed(SEED + 1)
    np.random.seed(SEED + 1)
    shapley = cast(Callable[..., Learned], namespace["PolicyShapley"])(env, args, char, train, validation)
    records = _record(shapley, train, validation, "sample", sampler=char.sampler)
    _run(namespace, shapley, args, records)
    return records


def _record(
    model: Learned, train: Buffer, validation: Buffer, method: str, sampler: Sampler | None = None
) -> dict[str, object]:
    """Wrap the buffers' and sampler's draws; exact-buffer reads (start=0) are not draws."""
    records: dict[str, object] = {
        "initial": [_numpy(parameter) for parameter in model.model.parameters()],
        "train_rows": [],
        "validation_rows": [],
        "coalitions": [],
    }
    for buffer, key in ((train, "train_rows"), (validation, "validation_rows")):
        original = buffer.sample
        log = cast(list[HostArray], records[key])

        def sample(
            batch_size: int,
            *keys: str,
            start: int | None = None,
            replace: bool = True,
            _buffer: Buffer = buffer,
            _original: Callable[..., list[Tensor]] = original,
            _log: list[HostArray] = log,
        ) -> list[Tensor]:
            if start is not None:
                return _original(batch_size, *keys, start=start, replace=replace)
            # Same call, same global stream as upstream Buffer.sample.
            index = np.random.choice(_buffer.size, batch_size, replace=replace)
            _log.append(index.copy())
            return [_buffer.buffer[name][index] for name in keys]

        buffer.sample = sample
    target = sampler if sampler is not None else model.sampler
    draw = getattr(target, method)
    coalitions = cast(list[HostArray], records["coalitions"])

    def recorded(*args: object, **kwargs: object) -> Tensor:
        value = cast(Tensor, draw(*args, **kwargs))
        coalitions.append(_numpy(value).astype(np.bool_))
        return value

    setattr(target, method, recorded)
    return records


def _run(namespace: dict[str, object], model: Learned, args: SimpleNamespace, records: dict[str, object]) -> None:
    writer = Writer()
    cast(Callable[..., None], namespace["train_models"])([model], args, writer, "parity")
    records["scalars"] = dict(writer.scalars)
    records["final"] = [_numpy(parameter) for parameter in model.model.parameters()]


def _records(prefix: str, records: dict[str, object]) -> dict[str, HostArray]:
    """Draws reshaped to (epoch, batch, ...); per-epoch logged scalars; initial and best-restored weights."""
    scalars = cast(dict[str, list[tuple[int, float]]], records["scalars"])
    train = cast(list[HostArray], records["train_rows"])
    validation = cast(list[HostArray], records["validation_rows"])
    coalitions = cast(list[HostArray], records["coalitions"])
    epochs = len(validation)
    batches = len(train) // epochs
    per_epoch = batches + 1  # Each epoch draws one coalition per batch, then one for validation.
    arrays: dict[str, HostArray] = {
        f"{prefix}_train_rows": np.asarray(train, np.int32).reshape(epochs, batches, -1),
        f"{prefix}_validation_rows": np.asarray(validation, np.int32),
    }
    grouped = [coalitions[epoch * per_epoch : (epoch + 1) * per_epoch] for epoch in range(epochs)]
    arrays[f"{prefix}_train_coalitions"] = np.asarray([group[:batches] for group in grouped], np.bool_)
    arrays[f"{prefix}_validation_coalitions"] = np.asarray([group[batches] for group in grouped], np.bool_)
    for tag, values in scalars.items():
        name = tag.removeprefix("model/").split("_", 1)[1]  # epoch_loss, val_loss, exact_loss
        arrays[f"{prefix}_{name}"] = np.asarray([value for _, value in values], np.float32)
        arrays[f"{prefix}_{name}_steps"] = np.asarray([step for step, _ in values], np.int32)
    for label in ("initial", "final"):
        for index, value in enumerate(cast(list[HostArray], records[label])):
            arrays[f"{prefix}_{label}_{index}"] = value
    return arrays


def _buffers(namespace: dict[str, object], env: SimpleNamespace, observations: HostArray) -> tuple[Buffer, Buffer]:
    cls = cast(Callable[..., Buffer], namespace["Buffer"])
    train, validation = cls(TRAIN_ROWS, env, "obs"), cls(VALIDATION_ROWS, env, "obs")
    train.add(obs=observations[:TRAIN_ROWS].astype(np.float32))
    validation.add(obs=observations[TRAIN_ROWS:].astype(np.float32))
    return train, validation


def _agent(
    namespace: dict[str, object], torch: Torch, env: object, *, features: int, actions: int, width: int
) -> Agent:
    cls = cast(type, namespace["DQN"])
    builder = cast(Builder, namespace["models"])
    torch.manual_seed(SEED)
    agent = object.__new__(cls)
    simple = builder.SimpleNN
    architecture = _architecture(features, width, softmax=False)
    vars(agent)["model"] = simple("DQN", (actions,), **copy.deepcopy(architecture))
    vars(agent)["target_critic"] = simple("DQN", (actions,), **copy.deepcopy(architecture))
    init = cast(Callable[..., None], cast(type, namespace["DQN_Super"]).__init__)
    init(agent, env, SimpleNamespace(start_e=0.2, learning_rate=LEARNING_RATE))
    return cast(Agent, agent)  # Source class adapter.


def _architecture(features: int, width: int, *, softmax: bool) -> dict[str, list[object]]:
    architecture: dict[str, list[object]] = {
        "input": ["Linear", [features, width]],
        "input_activation": ["ReLU", []],
        "hidden1": ["Linear", [width, width]],
        "hidden1_activation": ["ReLU", []],
        "output": ["Linear", [width, None]],
    }
    if softmax:
        architecture["output_activation"] = ["Softmax", [-1]]
    return architecture


def _namespace(root: Path, torch: Torch) -> dict[str, object]:
    package = root / "fastsverl/fastsverl"
    spec = importlib.util.spec_from_file_location("reference_models", package / "models.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load models.py")
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    namespace: dict[str, object] = {
        "torch": torch,
        "np": np,
        "copy": copy,
        "time": time,
        "random": importlib.import_module("random"),
        "itertools": itertools,
        "defaultdict": defaultdict,
        "namedtuple": importlib.import_module("collections").namedtuple,
        "comb": importlib.import_module("math").comb,
        "tqdm": _passthrough,
        "gymnasium": importlib.import_module("gymnasium"),
        "F": importlib.import_module("torch.nn.functional"),
        "Categorical": importlib.import_module("torch.distributions").Categorical,
        "DEVICE": torch.device("cpu"),
        "models": models,
        "SimpleNN": models.SimpleNN,
        "ImportanceWeightedMSELoss": models.ImportanceWeightedMSELoss,
    }
    for name, definitions in (
        ("utils.py", ("ShapleySampler", "Buffer")),
        ("dqn.py", ("DQN_Super", "DQN")),
        (
            "characteristics.py",
            (
                "Characteristic",
                "CharacteristicModel",
                "ValuePolicyCharacteristic",
                "ValuePolicyCharacteristicModel",
                "ValueCharacteristicModel",
                "PolicyCharacteristicModel",
            ),
        ),
        ("shapley.py", ("Shapley", "PolicyShapley")),
        ("training.py", ("get_data", "fill_buffer", "train_models")),
        ("envs/hypercube.py", ("Hypercube",)),
    ):
        path = package / name
        tree = ast.parse(path.read_text())
        selected: list[ast.stmt] = [
            node for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in definitions
        ]
        if len(selected) != len(definitions):
            raise RuntimeError(f"missing reference definitions in {name}")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)  # ruff: ignore[exec-builtin] -- verified pinned source, no external configuration code.
    return namespace


def _numpy(tensor: Tensor) -> HostArray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def _passthrough[T](iterable: T, *_: object, **__: object) -> T:
    return iterable


if __name__ == "__main__":
    main()
