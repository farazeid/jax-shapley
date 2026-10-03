# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy<2", "gymnasium==0.29.1", "tqdm"]
# ///
"""Capture upstream transition models and value iteration for outcome explanations.

Builds the unchanged `GWB` and `Mastermind` (2, 2, 2) environments, reads their
transition tables `P` (decoded to observations), and runs the unchanged
`value_iteration` under a uniform policy and the fixture agent's epsilon-greedy
policy with gamma = 1, the outcome characteristics' setting.
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
SOURCES = ("utils.py", "models.py", "dqn.py", "envs/gwb.py", "envs/mastermind.py")
SEED = 1729
EPSILON = 0.2
CASES = (("gridworld", 2, 4), ("mastermind222", 8, 4))


class Tensor(Protocol):
    data: Tensor

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
                "collector": "unchanged GWB/Mastermind get_P and value_iteration, CPU",
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
    return arrays, torch.__version__


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
    for name, definitions in (("utils.py", ("value_iteration",)), ("dqn.py", ("DQN_Super", "DQN"))):
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
