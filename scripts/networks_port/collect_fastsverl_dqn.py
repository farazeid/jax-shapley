# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy>=2.0"]
# ///
"""Capture pinned DQN numerical units without importing upstream telemetry.

Execute the unchanged DQN_Super AST with its numerical dependencies. Environment
and replay orchestration are tested separately; this collector controls batches
and weights instead of asserting identical Torch/JAX random streams.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import itertools
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed read-only git operations.
from collections import namedtuple
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import numpy as np

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"


class Tensor(Protocol):
    grad: Tensor | None

    def detach(self) -> Tensor: ...
    def cpu(self) -> Tensor: ...
    def numpy(self) -> FloatArray: ...
    def backward(self) -> None: ...
    def __add__(self, other: Tensor) -> Tensor: ...
    def __mul__(self, other: Tensor) -> Tensor: ...
    def __rmul__(self, other: float) -> Tensor: ...
    def __rsub__(self, other: int) -> Tensor: ...
    def max(self, dim: int) -> tuple[Tensor, Tensor]: ...
    def __getitem__(self, index: int) -> Tensor: ...


class ModelOutput(Protocol):
    DQN: Tensor


class ActionInfo(Protocol):
    logprob: Tensor


class Model(Protocol):
    def __call__(self, obs: Tensor) -> ModelOutput: ...
    def parameters(self) -> Iterator[Tensor]: ...


class Loss(Protocol):
    DQN_loss: Tensor


class Optimizer(Protocol):
    def zero_grad(self) -> None: ...


class Agent(Protocol):
    model: Model
    target_critic: Model
    optimizer: Optimizer
    epsilon: float

    def choose_action(self, obs: FloatArray, exp: bool, action: Tensor) -> ActionInfo: ...
    def pi_greedy(self, obs: Tensor) -> Tensor: ...
    def pi_explore(self, obs: Tensor) -> Tensor: ...
    def V(self, obs: Tensor) -> Tensor: ...  # ruff: ignore[invalid-function-name] -- source method.
    def forward(self, obs: Tensor, actions: Tensor, rewards: Tensor, n_obs: Tensor, terminations: Tensor) -> Loss: ...
    def backward(self, loss: Loss) -> Loss: ...
    def update_target(self) -> None: ...
    def linear_schedule(self, step: int) -> None: ...


class Torch(Protocol):
    __version__: str

    def manual_seed(self, seed: int) -> object: ...
    def set_num_threads(self, count: int) -> None: ...
    def from_numpy(self, values: FloatArray) -> Tensor: ...
    def tensor(self, values: list[int], *, dtype: object) -> Tensor: ...
    def device(self, name: str) -> object: ...

    long: object


class Builder(Protocol):
    def SimpleNN(self, name: str, output_shape: tuple[int, ...], **architecture: object) -> Model: ...  # ruff: ignore[invalid-function-name] -- source constructor.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, output = Path(args.source_root), Path(args.output)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required")
    commit = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
    if commit != REFERENCE_COMMIT:
        raise ValueError("unexpected reference commit")
    hashes: dict[str, str] = {}
    for name in ("dqn.py", "models.py"):
        relative = f"fastsverl/fastsverl/{name}"
        committed = subprocess.check_output([git, "-C", str(root), "show", f"HEAD:{relative}"])  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
        if committed != (root / relative).read_bytes():
            raise ValueError(f"modified reference {name}")
        hashes[name] = hashlib.sha256(committed).hexdigest()
    arrays, version = capture(root)
    np.savez_compressed(output, allow_pickle=False, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "fastsverl_commit": commit,
                "source_sha256": hashes,
                "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "torch_version": version,
                "seed": 1729,
                "gamma": 0.9,
                "tau": 0.01,
                "steps": 3,
                "collector": "unchanged DQN_Super AST, CPU, controlled batches",
            },
            indent=2,
        )
        + "\n"
    )


def capture(root: Path) -> tuple[dict[str, FloatArray], str]:
    torch = cast(Torch, importlib.import_module("torch"))  # Optional reference-only boundary.
    models = cast(Builder, _load(root / "fastsverl/fastsverl/models.py", "reference_models"))
    path = root / "fastsverl/fastsverl/dqn.py"
    source = path.read_text()
    tree = ast.parse(source)
    definitions: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("DQN_Super", "DQN")
    ]
    module = ModuleType("reference_dqn")
    module.__dict__.update(
        torch=torch,
        np=np,
        F=importlib.import_module("torch.nn.functional"),
        Categorical=importlib.import_module("torch.distributions").Categorical,
        itertools=itertools,
        namedtuple=namedtuple,  # ruff: ignore[collections-named-tuple] -- unchanged upstream runtime tuple factory.
        DEVICE=torch.device("cpu"),
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), module.__dict__)  # ruff: ignore[exec-builtin] -- verified pinned source, no external configuration code.
    cls = cast(type[Agent], module.__dict__["DQN"])  # Source class adapter.
    torch.set_num_threads(1)
    arrays: dict[str, FloatArray] = {}
    for name, features, width, actions in (
        ("gridworld", 2, 64, 4),
        ("mastermind222", 8, 64, 4),
        ("hypercube", 5, 128, 10),
    ):
        torch.manual_seed(1729)
        agent = object.__new__(cls)
        architecture = {
            "input": ["Linear", [features, width]],
            "relu0": ["ReLU", []],
            "hidden": ["Linear", [width, width]],
            "relu1": ["ReLU", []],
            "output": ["Linear", [width, actions]],
        }
        agent.model = models.SimpleNN("DQN", (actions,), **architecture)
        agent.target_critic = models.SimpleNN("DQN", (actions,), **architecture)
        args = SimpleNamespace(
            start_e=1.0,
            end_e=0.01,
            learning_rate=2.5e-4,
            gamma=0.9,
            tau=0.01,
            exploration_fraction=0.5,
            total_timesteps=100,
        )
        # Invoke unchanged initialization on the prepared models, without a replay buffer.
        module.__dict__["DQN_Super"].__init__(
            agent, SimpleNamespace(single_action_space=SimpleNamespace(n=actions)), args
        )
        obs = torch.from_numpy(np.arange(4 * features, dtype=np.float32).reshape(4, features) % 3)
        next_obs = torch.from_numpy(np.arange(4 * features, dtype=np.float32).reshape(4, features) % 4)
        action = torch.tensor([0, 1, 2, 0], dtype=torch.long)
        rewards = torch.from_numpy(np.array([-1, 0, 1, -1], np.float32))
        terminated = torch.from_numpy(np.array([0, 1, 0, 1], np.float32))
        for field, value in (
            ("observations", obs),
            ("next_observations", next_obs),
            ("actions", action),
            ("rewards", rewards),
            ("terminated", terminated),
        ):
            arrays[f"{name}_{field}"] = _array(value)
        arrays[f"{name}_greedy"] = _array(agent.pi_greedy(obs))
        arrays[f"{name}_logprob"] = _array(agent.choose_action(_array(obs), False, action).logprob)
        arrays[f"{name}_targets"] = _array(
            rewards + args.gamma * agent.target_critic(next_obs).DQN.max(dim=1)[0] * (1 - terminated)
        )
        agent.epsilon = 0.2
        arrays[f"{name}_explore"] = _array(agent.pi_explore(obs))
        for index, parameter in enumerate(agent.model.parameters()):
            arrays[f"{name}_initial_{index}"] = _array(parameter)
        for step in range(3):
            loss = agent.forward(obs, action, rewards, next_obs, terminated)
            arrays[f"{name}_loss_{step}"] = _array(loss.DQN_loss)
            agent.optimizer.zero_grad()
            loss.DQN_loss.backward()
            for index, parameter in enumerate(agent.model.parameters()):
                if parameter.grad is None:
                    raise RuntimeError("reference gradient missing")
                arrays[f"{name}_gradient_{step}_{index}"] = _array(parameter.grad)
            agent.backward(agent.forward(obs, action, rewards, next_obs, terminated))
            agent.update_target()
            for index, parameter in enumerate(agent.model.parameters()):
                arrays[f"{name}_updated_{step}_{index}"] = _array(parameter)
            for index, parameter in enumerate(agent.target_critic.parameters()):
                arrays[f"{name}_target_{step}_{index}"] = _array(parameter)
        for step in (0, 25, 50, 100):
            agent.linear_schedule(step)
            arrays[f"epsilon_{step}"] = np.asarray(agent.epsilon, np.float32)
    return arrays, torch.__version__


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _array(tensor: Tensor) -> FloatArray:
    return tensor.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    main()
