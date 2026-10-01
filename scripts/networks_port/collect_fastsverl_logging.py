# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy>=2.0"]
# ///
"""Execute the unchanged pinned training-loop AST with controlled event inputs.

Run in the separate reference environment, not the project test environment.
The fake agent/environment isolate logging selection from random trajectories.
"""

import argparse
import ast
import hashlib
import importlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Protocol, cast

import numpy as np
from numpy.typing import NDArray

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
SOURCE_SHA256 = "f9f47faa0bb04faa08eee6dbfff2006d617656c4b12e46b74ea0d2cf16392e77"


@dataclass(frozen=True, slots=True)
class ScalarRecord:
    name: str
    value: float
    step: int


@dataclass
class ReferenceWriter:
    records: list[ScalarRecord] = field(default_factory=list)

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.records.append(ScalarRecord(name, float(value), int(step)))


class Loss(NamedTuple):
    DQN_loss: float


class ReferenceAgent:
    def __init__(self) -> None:
        self.epsilon = 0.5
        self.step = 0
        self.args = SimpleNamespace()
        self.buffer = SimpleNamespace(add=_discard)
        self.model = SimpleNamespace(eval=lambda: None)

    def linear_schedule(self, step: int) -> None:
        self.step = step

    def choose_action(self, observations: NDArray[np.int32]) -> SimpleNamespace:
        del observations
        return SimpleNamespace(action=np.zeros(2, np.int32), logprob=np.zeros(2, np.float32))

    def update(self, *values: object) -> Loss:
        del values
        return Loss(self.step + 0.25)

    def update_target(self) -> None:
        pass


class ReferenceEnv:
    num_envs: int = 2

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[NDArray[np.int32], dict[str, object]]:
        del seed
        return np.zeros((2, 2), np.int32), {}

    def step(
        self, actions: NDArray[np.int32]
    ) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.bool_], NDArray[np.bool_], dict[str, object]]:
        del actions
        index = self.step_count
        self.step_count += 1
        completed = [
            {"episode": {"r": float(index + env), "l": index + env + 1}} if index % (7 if env == 0 else 11) == 0 else {}
            for env in range(2)
        ]
        return (
            np.zeros((2, 2), np.int32),
            np.zeros(2, np.float32),
            np.zeros(2, bool),
            np.zeros(2, bool),
            {"final_info": completed, "final_observation": np.zeros((2, 2), np.int32)},
        )


class Trainer(Protocol):
    def __call__(
        self,
        agent: ReferenceAgent,
        updates: list[object],
        env: ReferenceEnv,
        args: SimpleNamespace,
        writer: ReferenceWriter,
        seed: int,
        name: str,
    ) -> None: ...


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Pinned fastsverl/training.py")
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/fastsverl_logging.json"))
    options = parser.parse_args()
    source = Path(options.source).read_text()
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError("training.py does not match the pinned reference source")
    nodes: list[ast.stmt] = [
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "train_dqn"
    ]
    clock = iter([0.0, *([2.0] * 100)])
    namespace: dict[str, object] = {
        "torch": importlib.import_module("torch"),
        "time": SimpleNamespace(time=lambda: next(clock)),
        "tqdm": _progress,
        "get_data": _empty_data,
        "evaluate_dqn": _evaluation,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned training.py", "exec"), namespace)  # ruff: ignore[exec-builtin] -- exact hash-verified reference source, isolated collector.
    trainer = cast(
        Trainer, namespace["train_dqn"]
    )  # The hash-verified upstream function has this controlled interface.
    args = SimpleNamespace(
        total_timesteps=108,
        learning_starts=0,
        train_frequency=3,
        target_network_frequency=1,
        batch_size=3,
        verbose=0,
        eval_episodes=2,
    )
    writer = ReferenceWriter()
    trainer(ReferenceAgent(), [], ReferenceEnv(), args, writer, 1, "unused")
    from dataclasses import asdict

    Path(options.output).write_text(
        json.dumps(
            {
                "reference_commit": REFERENCE_COMMIT,
                "source_sha256": SOURCE_SHA256,
                "torch_version": version("torch"),
                "num_envs": 2,
                "vector_steps": 54,
                "records": [asdict(record) for record in writer.records],
            },
            indent=2,
        )
        + "\n"
    )


def _discard(**values: object) -> None:
    del values


def _progress(sequence: Iterable[int], *values: object) -> Iterable[int]:
    del values
    return sequence


def _empty_data(*values: object) -> tuple[()]:
    del values
    return ()


def _evaluation(*values: object) -> list[float]:
    del values
    return [4.0, 5.0, 6.0]


if __name__ == "__main__":
    main()
