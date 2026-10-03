# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy<2", "gymnasium==0.29.1", "tqdm"]
# ///
"""Capture upstream transition models, value iteration and outcome explanation units.

Builds the unchanged `GWB` and `Mastermind` (2, 2, 2) environments, reads their
transition tables `P` (decoded to observations), and runs the unchanged
`value_iteration` under a uniform policy and the fixture agent's epsilon-greedy
policy with gamma = 1, the outcome characteristics' setting.

The outcome units run the unchanged `PerformanceCharacteristicOffPolicy`,
`PerformanceCharacteristicOnPolicy` and `PerformanceShapley` on fixed batches and
coalitions, explaining the explanation fixture's epsilon-greedy agent and trained
behaviour characteristic. (The agent's greedy Gridworld policy loops forever, so
the source's undiscounted value iteration would never converge under it.) The next actions the off-policy target samples are recorded so
they can be replayed.
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
from collections import defaultdict, namedtuple
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import numpy as np

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
SOURCES = ("utils.py", "models.py", "dqn.py", "characteristics.py", "shapley.py", "envs/gwb.py", "envs/mastermind.py")
STEPS, BATCH_SIZE, LEARNING_RATE = 3, 8, 2.5e-4
TAU = 0.01
SEED = 1729
EPSILON = 0.2
CASES = (("gridworld", 2, 4), ("mastermind222", 8, 4))


class Tensor(Protocol):
    data: Tensor
    grad: Tensor | None

    @property
    def shape(self) -> tuple[int, ...]: ...
    def __getitem__(self, index: slice) -> Tensor: ...
    def detach(self) -> Tensor: ...
    def cpu(self) -> Tensor: ...
    def numpy(self) -> FloatArray: ...
    def copy_(self, other: Tensor) -> Tensor: ...


class Torch(Protocol):
    __version__: str

    def manual_seed(self, seed: int) -> object: ...
    def set_num_threads(self, count: int) -> None: ...
    def from_numpy(self, values: HostArray) -> Tensor: ...
    def device(self, name: str) -> object: ...

    Tensor: Callable[[HostArray], Tensor]


class Model(Protocol):
    def parameters(self) -> Iterator[Tensor]: ...


class Agent(Protocol):
    model: Model
    epsilon: float

    def pi_explore(self, obs: Tensor) -> Tensor: ...


class Sampler(Protocol):
    sample: Callable[..., Tensor]
    sample_rand: Callable[..., Tensor]


class Learned(Protocol):
    model: Model
    sampler: Sampler


class Outcome(Learned, Protocol):
    target_critic: Model
    with_exact_char: bool
    exact: dict[tuple[int, ...], dict[tuple[float, ...], Tensor]]

    def update(self, *inputs: Tensor) -> object: ...
    def update_target(self) -> None: ...
    def get_val(self, e_obs: Tensor, coalitions: Tensor) -> Tensor: ...
    def null(self, obs: Tensor) -> Tensor: ...
    def get_exact(self) -> None: ...
    def exact_loss(self) -> object: ...


class Explainer(Protocol):
    model: Model
    with_exact_char: bool
    exact: dict[tuple[float, ...], Tensor]

    def update(self, obs: Tensor) -> object: ...
    def sv(self, obs: Tensor) -> Tensor: ...
    def get_exact(self) -> None: ...
    def exact_loss(self) -> object: ...


class ExactBuffer:
    """Sequential read-only stand-in for the upstream Buffer, as used by exact computations."""

    def __init__(self, observations: Tensor) -> None:
        self.observations = observations
        self.size = int(observations.shape[0])

    def sample(self, batch_size: int, *keys: str, start: int | None = None) -> list[Tensor]:
        if keys != ("obs",) or start is None or start + batch_size > self.size:
            raise ValueError("exact buffers are read sequentially as observations only")
        return [self.observations[start : start + batch_size]]


class Environment(Protocol):
    P: dict[int, object]

    def decode(self, state: int) -> HostArray: ...


class Builder(Protocol):
    def SimpleNN(self, name: str, output_shape: tuple[int, ...], **architecture: object) -> Model: ...  # ruff: ignore[invalid-function-name] -- source constructor.


class SaveArrays(Protocol):
    def __call__(self, file: Path, /, **arrays: HostArray) -> None: ...


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
    arrays, version = capture(root, reference)
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
                "epsilon": EPSILON,
                "gamma": 1.0,
                "cases": {name: {"features": features, "actions": actions} for name, features, actions in CASES},
                "steps": STEPS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "tau": TAU,
                "collector": "unchanged GWB/Mastermind get_P, value_iteration and outcome ASTs, CPU, recorded draws",
            },
            indent=2,
        )
        + "\n"
    )


def capture(root: Path, reference: dict[str, HostArray]) -> tuple[dict[str, HostArray], str]:
    torch = cast(Torch, importlib.import_module("torch"))  # Optional reference-only boundary.
    torch.set_num_threads(1)
    namespace = _namespace(root, torch)
    arrays: dict[str, HostArray] = {}
    np.random.seed(SEED)  # Mastermind draws its goal order from the legacy global stream.
    environments = {
        "gridworld": cast(Callable[[], Environment], namespace["GWB"])(),
        "mastermind222": cast(Callable[..., Environment], namespace["Mastermind"])(
            code_size=2, num_guesses=2, num_pegs=2
        ),
    }
    cast(Callable[[], None], getattr(environments["mastermind222"], "get_P"))()  # ruff: ignore[get-attr-with-constant] -- source method.
    for name, features, actions in CASES:
        env = environments[name]
        arrays.update(transitions(name, env))
        agent = _agent(namespace, torch, name, features, actions, reference)
        value_iteration = cast(Callable[..., dict[tuple[int, ...], HostArray]], namespace["value_iteration"])
        uniform = np.full(actions, 1.0 / actions)
        policies: dict[str, Callable[[HostArray], HostArray]] = {
            "uniform": lambda observation, uniform=uniform: uniform,
            "explore": lambda observation, agent=agent: agent.pi_explore(torch.Tensor(observation)).numpy(),
        }
        for label, policy in policies.items():
            table = value_iteration(SimpleNamespace(unwrapped=env), gamma=1.0, policy=policy)
            states = sorted(table)
            arrays[f"{name}_q_{label}_states"] = np.asarray(states, np.int32)
            arrays[f"{name}_q_{label}"] = np.asarray([table[state] for state in states], np.float32)
        arrays.update(capture_units(namespace, torch, name, env, agent, reference, arrays))
    return arrays, torch.__version__


def capture_units(
    namespace: dict[str, object],
    torch: Torch,
    name: str,
    env: Environment,
    agent: Agent,
    reference: dict[str, HostArray],
    tables: dict[str, HostArray],
) -> dict[str, HostArray]:
    """Off- and on-policy outcome characteristics and the outcome Shapley model on fixed inputs."""
    features, actions = reference[f"{name}_unique"].shape[1], tables[f"{name}_p_probability"].shape[1]
    stub = SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(features,)),
        single_action_space=SimpleNamespace(n=actions, shape=()),
        envs=[SimpleNamespace(unwrapped=env)],
        unwrapped=SimpleNamespace(get_attr=lambda attribute: [env]),
    )
    observations = reference[f"{name}_exact_buffer"].astype(np.float32)
    buffer = ExactBuffer(torch.from_numpy(observations))
    behaviour = _behaviour(namespace, torch, stub, agent, buffer, name, features, reference)
    batch = _transitions(tables, name, observations, actions)
    arrays: dict[str, HostArray] = {f"{name}_batch_{key}": value for key, value in batch.items()}
    unique = torch.from_numpy(reference[f"{name}_unique"].astype(np.float32))
    probe = _coalitions(int(unique.shape[0]), features, 7)
    arrays[f"{name}_probe_coalitions"] = probe
    tensors = {key: torch.from_numpy(value) for key, value in batch.items()}
    tensors["action"] = torch.from_numpy(batch["action"].astype(np.int64))
    sampled: list[HostArray] = []
    namespace["Categorical"] = _recording_categorical(sampled)

    off = _outcome(namespace, torch, "PerformanceCharacteristicOffPolicy", stub, agent, behaviour, features)
    arrays.update(_parameters(f"{name}_off_initial", off.model))
    for step in range(STEPS):
        coalitions = _coalitions(BATCH_SIZE, features, step)
        arrays[f"{name}_off_coalitions_{step}"] = coalitions
        off.sampler.sample_rand = _fixed(torch.from_numpy(coalitions.astype(np.float32)))
        sampled.clear()
        loss = off.update(
            tensors["obs"],
            tensors["action"],
            tensors["reward"],
            tensors["next_obs"],
            tensors["terminated"],
            tensors["e_obs"],
        )
        arrays[f"{name}_off_next_actions_{step}"] = np.asarray(sampled[0], np.int32)
        arrays[f"{name}_off_loss_{step}"] = _numpy(cast(tuple[Tensor, ...], loss)[0])
        if step == 0:
            arrays.update(_gradients(f"{name}_off_gradient", off.model))
    arrays.update(_parameters(f"{name}_off_final", off.model))
    arrays.update(_parameters(f"{name}_off_target", off.target_critic))
    arrays.update(_values(f"{name}_off", off, unique, torch.from_numpy(probe.astype(np.float32))))
    for label, flag in (("with_exact_char", True), ("without_exact_char", False)):
        off.with_exact_char = flag
        vars(off).pop("exact", None)
        loss = off.exact_loss()
        arrays[f"{name}_exact_{label}"] = _exact_table(off, unique, features)
        arrays[f"{name}_off_exact_loss_{label}"] = _numpy(cast(tuple[Tensor, ...], loss)[0])

    on = _outcome(namespace, torch, "PerformanceCharacteristicOnPolicy", stub, agent, behaviour, features)
    arrays.update(_parameters(f"{name}_on_initial", on.model))
    for step in range(STEPS):
        coalitions = torch.from_numpy(_coalitions(BATCH_SIZE, features, step).astype(np.float32))
        loss = on.update(
            tensors["obs"], tensors["reward"], tensors["next_obs"], tensors["terminated"], tensors["e_obs"], coalitions
        )
        on.update_target()
        arrays[f"{name}_on_loss_{step}"] = _numpy(cast(tuple[Tensor, ...], loss)[0])
        if step == 0:
            arrays.update(_gradients(f"{name}_on_gradient", on.model))
    arrays.update(_parameters(f"{name}_on_final", on.model))
    arrays.update(_parameters(f"{name}_on_target", on.target_critic))
    arrays.update(_values(f"{name}_on", on, unique, torch.from_numpy(probe.astype(np.float32))))
    for label, flag in (("with_exact_char", True), ("without_exact_char", False)):
        on.with_exact_char = flag
        vars(on).pop("exact", None)
        arrays[f"{name}_on_exact_loss_{label}"] = _numpy(cast(tuple[Tensor, ...], on.exact_loss())[0])

    # The Shapley model explains the off-policy characteristic, whose ground truth uses the exact behaviour.
    off.with_exact_char = True
    vars(off).pop("exact", None)
    off.get_exact()
    arrays.update(_shapley(namespace, torch, name, stub, off, features, tensors["obs"], unique))
    return arrays


def _shapley(
    namespace: dict[str, object],
    torch: Torch,
    name: str,
    stub: SimpleNamespace,
    char: Outcome,
    features: int,
    batch: Tensor,
    unique: Tensor,
) -> dict[str, HostArray]:
    arguments = SimpleNamespace(
        model_arch=_architecture(features, softmax=False),
        learning_rate=LEARNING_RATE,
        with_exact_char=True,
        verbose=False,
    )
    torch.manual_seed(SEED + 3)
    shapley = cast(Callable[..., Explainer], namespace["PerformanceShapley"])(stub, arguments, char)
    arrays = _parameters(f"{name}_shapley_initial", shapley.model)
    for step in range(STEPS):
        coalition = (np.arange(features) + step) % 2 == 0
        arrays[f"{name}_shapley_coalition_{step}"] = coalition
        char.sampler.sample = _fixed(torch.from_numpy(coalition[None].astype(np.float32)))
        loss = shapley.update(batch)
        arrays[f"{name}_shapley_loss_{step}"] = _numpy(cast(tuple[Tensor, ...], loss)[0])
        if step == 0:
            arrays.update(_gradients(f"{name}_shapley_gradient", shapley.model))
    arrays.update(_parameters(f"{name}_shapley_final", shapley.model))
    arrays[f"{name}_shapley_sv_unique"] = _numpy(shapley.sv(unique))
    for label, flag in (("with_exact_char", True), ("without_exact_char", False)):
        shapley.with_exact_char = flag
        vars(shapley).pop("exact", None)
        shapley.get_exact()
        keys = [tuple(float(value) for value in row) for row in _numpy(unique)]
        arrays[f"{name}_shapley_exact_{label}"] = np.stack([_numpy(shapley.exact[key]) for key in keys])
        arrays[f"{name}_shapley_exact_loss_{label}"] = _numpy(cast(tuple[Tensor, ...], shapley.exact_loss())[0])
    return arrays


def _behaviour(
    namespace: dict[str, object],
    torch: Torch,
    stub: SimpleNamespace,
    agent: Agent,
    buffer: ExactBuffer,
    name: str,
    features: int,
    reference: dict[str, HostArray],
) -> Learned:
    arguments = SimpleNamespace(
        model_arch=_architecture(features, softmax=True), learning_rate=LEARNING_RATE, policy="explore", verbose=False
    )
    cls = cast(Callable[..., Learned], namespace["PolicyCharacteristicModel"])
    behaviour = cls(agent, stub, arguments, buffer, None, buffer)
    for index, parameter in enumerate(behaviour.model.parameters()):
        parameter.data.copy_(torch.from_numpy(reference[f"{name}_char_behaviour_final_{index}"]))
    return behaviour


def _outcome(
    namespace: dict[str, object],
    torch: Torch,
    cls: str,
    stub: SimpleNamespace,
    agent: Agent,
    behaviour: Learned,
    features: int,
) -> Outcome:
    arguments = SimpleNamespace(
        model_arch=_architecture(3 * features, softmax=False),
        learning_rate=LEARNING_RATE,
        gamma=1.0,
        tau=TAU,
        start_e=None,
        verbose=False,
        with_exact_char=True,
        buffer_size=BATCH_SIZE,
    )
    torch.manual_seed(SEED + 2)
    return cast(Callable[..., Outcome], namespace[cls])(agent, stub, arguments, behaviour)


def _values(prefix: str, model: Outcome, unique: Tensor, coalitions: Tensor) -> dict[str, HostArray]:
    """get_val at each unique state with a fixed coalition per row, and the null value."""
    return {
        f"{prefix}_get_val": _numpy(model.get_val(unique, coalitions)),
        f"{prefix}_null": _numpy(model.null(unique)),
    }


def _exact_table(model: Outcome, unique: Tensor, features: int) -> HostArray:
    """Exact outcome values as (coalitions, unique states) in cartesian_prod order."""
    states = [tuple(float(value) for value in row) for row in _numpy(unique)]
    coalitions = list(itertools.product((0, 1), repeat=features))
    return np.asarray(
        [[_numpy(model.exact[coalition][state])[0] for state in states] for coalition in coalitions], np.float32
    )


def _transitions(
    tables: dict[str, HostArray], name: str, observations: HostArray, actions: int
) -> dict[str, HostArray]:
    """A fixed batch of transitions read from P (first non-zero outcome), with independent e_obs rows."""
    states = [tuple(row) for row in tables[f"{name}_p_states"].tolist()]
    obs = observations[:BATCH_SIZE]
    action = np.arange(BATCH_SIZE) % actions
    following, reward, done = [], [], []
    for row, chosen in zip(obs.astype(np.int32).tolist(), action, strict=True):
        index = states.index(tuple(row))
        outcome = int(np.flatnonzero(tables[f"{name}_p_probability"][index, chosen] > 0)[0])
        following.append(tables[f"{name}_p_next"][index, chosen, outcome])
        reward.append(tables[f"{name}_p_reward"][index, chosen, outcome])
        done.append(tables[f"{name}_p_done"][index, chosen, outcome])
    return {
        "obs": obs.astype(np.float32),
        "action": action.astype(np.int32),
        "reward": np.asarray(reward, np.float32),
        "next_obs": np.asarray(following, np.float32),
        "terminated": np.asarray(done, np.float32),
        "e_obs": np.roll(observations, 3, axis=0)[:BATCH_SIZE].astype(np.float32),
    }


def _coalitions(rows: int, features: int, step: int) -> HostArray:
    grid_rows, columns = np.meshgrid(np.arange(rows), np.arange(features), indexing="ij")
    return ((grid_rows * 3 + columns * 5 + step * 7) % 5) < 2 + step % 2


def _recording_categorical(log: list[HostArray]) -> type:
    base = cast(type, importlib.import_module("torch.distributions").Categorical)

    class Recording(base):  # ty: ignore[unsupported-base] -- dynamic Torch distribution class.
        def sample(self, *arguments: object, **keywords: object) -> Tensor:
            value = cast(Tensor, super().sample(*arguments, **keywords))
            log.append(np.asarray(value.detach().cpu().numpy()))
            return value

    return Recording


def _architecture(features: int, *, softmax: bool) -> dict[str, list[object]]:
    architecture: dict[str, list[object]] = {
        "input": ["Linear", [features, 64]],
        "input_activation": ["ReLU", []],
        "hidden1": ["Linear", [64, 64]],
        "hidden1_activation": ["ReLU", []],
        "output": ["Linear", [64, None]],
    }
    if softmax:
        architecture["output_activation"] = ["Softmax", [-1]]
    return architecture


def _parameters(name: str, model: Model) -> dict[str, HostArray]:
    return {f"{name}_{index}": _numpy(parameter) for index, parameter in enumerate(model.parameters())}


def _gradients(name: str, model: Model) -> dict[str, HostArray]:
    arrays: dict[str, HostArray] = {}
    for index, parameter in enumerate(model.parameters()):
        if parameter.grad is None:
            raise RuntimeError("reference gradient missing")
        arrays[f"{name}_{index}"] = _numpy(parameter.grad)
    return arrays


def _fixed(value: Tensor) -> Callable[..., Tensor]:
    def sample(*_: object, **__: object) -> Tensor:
        return value

    return sample


def _numpy(tensor: Tensor) -> FloatArray:
    return tensor.detach().cpu().numpy().astype(np.float32)


def transitions(name: str, env: Environment) -> dict[str, HostArray]:
    """Decoded P: per state and action, padded (probability, next, reward, done) outcomes."""
    states = sorted(env.P)
    outcomes: list[list[list[tuple[float, HostArray, float, bool]]]] = []
    for state in states:
        per_action: list[list[tuple[float, HostArray, float, bool]]] = []
        entries = env.P[state]
        actions = entries.items() if isinstance(entries, dict) else enumerate(cast(list[object], entries))
        for _, transitions in sorted(actions, key=lambda item: item[0]):
            rows = [tuple(row) for row in cast(list[list[object]], transitions)]
            per_action.append([
                (
                    float(cast(float, row[0])),
                    env.decode(int(cast(int, row[1]))),
                    float(cast(float, row[2])),
                    bool(row[3]),
                )
                for row in rows
            ])
        outcomes.append(per_action)
    width = max(len(entries) for per_action in outcomes for entries in per_action)
    shape = (len(states), len(outcomes[0]), width)
    features = len(env.decode(states[0]))
    probability, reward = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    done, following = np.zeros(shape, np.bool_), np.zeros((*shape, features), np.int32)
    for s, per_action in enumerate(outcomes):
        for a, entries in enumerate(per_action):
            for k, (p, observation, r, terminal) in enumerate(entries):
                probability[s, a, k], reward[s, a, k], done[s, a, k] = p, r, terminal
                following[s, a, k] = observation
    return {
        f"{name}_p_states": np.asarray([env.decode(state) for state in states], np.int32),
        f"{name}_p_probability": probability,
        f"{name}_p_next": following,
        f"{name}_p_reward": reward,
        f"{name}_p_done": done,
    }


def _agent(
    namespace: dict[str, object], torch: Torch, name: str, features: int, actions: int, reference: dict[str, HostArray]
) -> Agent:
    cls = cast(type, namespace["DQN"])
    builder = cast(Builder, namespace["models"])
    agent = object.__new__(cls)
    architecture: dict[str, list[object]] = {
        "input": ["Linear", [features, 64]],
        "input_activation": ["ReLU", []],
        "hidden1": ["Linear", [64, 64]],
        "hidden1_activation": ["ReLU", []],
        "output": ["Linear", [64, None]],
    }
    torch.manual_seed(SEED)
    vars(agent)["model"] = builder.SimpleNN("DQN", (actions,), **copy.deepcopy(architecture))
    vars(agent)["target_critic"] = builder.SimpleNN("DQN", (actions,), **copy.deepcopy(architecture))
    init = cast(Callable[..., None], cast(type, namespace["DQN_Super"]).__init__)
    init(
        agent,
        SimpleNamespace(single_action_space=SimpleNamespace(n=actions)),
        SimpleNamespace(start_e=EPSILON, learning_rate=2.5e-4),
    )
    typed = cast(Agent, agent)  # Source class adapter.
    for index, parameter in enumerate(typed.model.parameters()):
        parameter.data.copy_(torch.from_numpy(reference[f"{name}_agent_{index}"]))
    return typed


def _namespace(root: Path, torch: Torch) -> dict[str, object]:
    package = root / "fastsverl/fastsverl"
    namespace: dict[str, object] = {
        "torch": torch,
        "np": np,
        "itertools": itertools,
        "defaultdict": defaultdict,
        "namedtuple": namedtuple,  # ruff: ignore[collections-named-tuple] -- unchanged upstream runtime tuple factory.
        "F": importlib.import_module("torch.nn.functional"),
        "Categorical": importlib.import_module("torch.distributions").Categorical,
        "DEVICE": torch.device("cpu"),
    }
    models = _load(package / "models.py", "reference_models")
    namespace.update(models=models, SimpleNN=models.SimpleNN)
    for name, module in (("envs/gwb.py", "reference_gwb"), ("envs/mastermind.py", "reference_mastermind")):
        loaded = _load(package / name, module)
        namespace.update({key: getattr(loaded, key) for key in ("GWB", "Mastermind") if hasattr(loaded, key)})
    namespace.update(
        comb=importlib.import_module("math").comb,
        tqdm=lambda iterable, *_, **__: iterable,
        ImportanceWeightedMSELoss=models.ImportanceWeightedMSELoss,
        MultiHeadNN=models.MultiHeadNN,
    )
    for name, definitions in (
        ("utils.py", ("value_iteration", "ShapleySampler", "Buffer")),
        ("dqn.py", ("DQN_Super", "DQN")),
        (
            "characteristics.py",
            (
                "Characteristic",
                "CharacteristicModel",
                "ValuePolicyCharacteristic",
                "ValuePolicyCharacteristicModel",
                "PolicyCharacteristicModel",
                "PerformanceCharacteristic",
                "PerformanceCharacteristicOnPolicy",
                "PerformanceCharacteristicOffPolicy",
            ),
        ),
        ("shapley.py", ("Shapley", "PerformanceShapley")),
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


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
