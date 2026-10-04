# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy<2"]
# ///
"""Capture the unchanged sampled characteristic (`ValuePolicyCharacteristicSample`).

Runs the source's `get_val` and `null` over a fixed pool held in the unchanged
`Buffer`, for observation batches and coalitions that include rows with no match.
Each call seeds NumPy (the pool permutation `Buffer.sample` draws) and torch (the
match scores `torch.rand` draws); replaying the same seeds records the permuted pool
and the scores, so a JAX port can select rows from identical inputs. v_F is a fixed
softmax-linear map, standing in for the explained policy.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed read-only git operations.
from collections import defaultdict, namedtuple
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import numpy as np

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
SOURCES = ("characteristics.py", "utils.py")
SEED = 4242
POOL, FEATURES, VALUES, ROWS, CALLS = 48, 4, 3, 12, 8


class SaveArrays(Protocol):
    def __call__(self, file: Path, /, **arrays: HostArray) -> None: ...


class Tensor(Protocol):
    def numpy(self) -> HostArray: ...
    def __matmul__(self, other: Tensor) -> Tensor: ...


class Torch(Protocol):
    __version__: str
    float32: object

    def use_deterministic_algorithms(self, mode: bool) -> None: ...
    def manual_seed(self, seed: int) -> object: ...
    def rand(self, shape: tuple[int, ...]) -> Tensor: ...
    def tensor(self, values: HostArray, dtype: object = None) -> Tensor: ...
    def as_tensor(self, values: HostArray, dtype: object = None) -> Tensor: ...
    def softmax(self, values: Tensor, dim: int) -> Tensor: ...


class Pool(Protocol):
    """The unchanged Buffer, built without environments: only `buffer` and `size` are read."""

    buffer: dict[str, Tensor]
    size: int


class Sampled(Protocol):
    """The unchanged ValuePolicyCharacteristicSample, with the attributes get_val and null read."""

    train_buffer: Pool
    pool_size: int
    args: SimpleNamespace
    v_F: Callable[[HostArray], Tensor]  # ruff: ignore[mixed-case-variable-in-class-scope] -- the source attribute name.

    def get_val(self, e_obs: Tensor, coalitions: Tensor, /) -> Tensor: ...
    def null(self, obs: Tensor) -> Tensor: ...


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
    for name in SOURCES:
        relative = f"fastsverl/fastsverl/{name}"
        committed = subprocess.check_output([git, "-C", str(root), "show", f"HEAD:{relative}"])  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
        if committed != (root / relative).read_bytes():
            raise ValueError(f"modified reference {name}")
        hashes[name] = hashlib.sha256(committed).hexdigest()
    arrays, version = capture(root)
    # NumPy < 2 would store an allow_pickle keyword as an array.
    cast(SaveArrays, np.savez_compressed)(output, **arrays)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "fastsverl_commit": commit,
                "source_sha256": hashes,
                "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "torch_version": version,
                "seed": SEED,
                "pool": POOL,
                "features": FEATURES,
                "rows": ROWS,
                "calls": CALLS,
                "num_samples": 1,
                "collector": "unchanged ValuePolicyCharacteristicSample.get_val/null and Buffer.sample ASTs, CPU",
            },
            indent=2,
        )
        + "\n"
    )


def capture(root: Path) -> tuple[dict[str, HostArray], str]:
    torch = cast(Torch, importlib.import_module("torch"))  # Typed boundary for the pinned torch API used here.
    torch.use_deterministic_algorithms(True)
    namespace = _namespace(root, torch)
    rng = np.random.default_rng(SEED)
    # Small-alphabet observations so coalitions match several pool rows; duplicates included.
    pool = rng.integers(-1, 3, size=(POOL, FEATURES)).astype(np.float32)
    pool[POOL // 2 :] = pool[: POOL // 2][rng.permutation(POOL // 2)]
    weights = torch.tensor(rng.normal(size=(FEATURES, VALUES)), dtype=torch.float32)

    def v_f(observations: HostArray) -> Tensor:
        return torch.softmax(torch.as_tensor(observations, dtype=torch.float32) @ weights, dim=-1)

    # Source classes are built without environments; only the attributes below are read.
    buffer = cast(Pool, object.__new__(cast(type, namespace["Buffer"])))
    buffer.buffer = {"obs": torch.tensor(pool)}
    buffer.size = POOL
    characteristic = cast(Sampled, object.__new__(cast(type, namespace["ValuePolicyCharacteristicSample"])))
    characteristic.train_buffer = buffer
    characteristic.pool_size = POOL
    characteristic.args = SimpleNamespace(num_samples=1)
    characteristic.v_F = v_f
    names = ("observations", "coalitions", "pools", "scores", "values", "pool_values")
    records: dict[str, list[HostArray]] = {name: [] for name in names}
    for call in range(CALLS):
        # Half the rows come from the pool (always matched); the rest are fresh (sometimes unmatched).
        observations = np.concatenate([
            pool[rng.choice(POOL, ROWS // 2, replace=False)],
            rng.integers(-1, 4, size=(ROWS - ROWS // 2, FEATURES)).astype(np.float32),
        ])
        coalition = rng.random(FEATURES) < 0.6
        if call == 0:
            coalition[:] = True  # The grand coalition: exact matches only.
        if call == 1:
            coalition[:] = False  # The empty coalition: every pool row matches.
        coalitions = np.broadcast_to(coalition, observations.shape).astype(np.float32)
        np.random.seed(SEED + call)
        index = np.random.choice(POOL, POOL, replace=False)
        torch.manual_seed(SEED + call)
        scores = torch.rand((ROWS, POOL)).numpy()
        np.random.seed(SEED + call)
        torch.manual_seed(SEED + call)
        values = characteristic.get_val(torch.tensor(observations), torch.tensor(coalitions))
        records["observations"].append(observations.astype(np.int32))
        records["coalitions"].append(coalition)
        records["pools"].append(pool[index].astype(np.int32))
        records["scores"].append(scores.astype(np.float32))
        records["values"].append(values.numpy().astype(np.float32))
        records["pool_values"].append(v_f(pool[index]).numpy().astype(np.float32))
    arrays: dict[str, HostArray] = {name: np.stack(values) for name, values in records.items()}
    arrays["null"] = characteristic.null(torch.tensor(pool)).numpy().astype(np.float32)
    arrays["pool"] = pool.astype(np.int32)
    return arrays, str(torch.__version__)


def _namespace(root: Path, torch: Torch) -> dict[str, object]:
    package = root / "fastsverl/fastsverl"
    namespace: dict[str, object] = {"torch": torch, "np": np, "defaultdict": defaultdict, "namedtuple": namedtuple}  # ruff: ignore[collections-named-tuple] -- unchanged upstream runtime tuple factory.
    for name, definitions in (
        ("utils.py", ("Buffer",)),
        ("characteristics.py", ("Characteristic", "ValuePolicyCharacteristic", "ValuePolicyCharacteristicSample")),
    ):
        path = package / name
        tree = ast.parse(path.read_text())
        selected: list[ast.stmt] = [
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in definitions
        ]
        if len(selected) != len(definitions):
            raise RuntimeError(f"missing reference definitions in {name}")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)  # ruff: ignore[exec-builtin] -- verified pinned source, no external configuration code.
    return namespace


if __name__ == "__main__":
    main()
