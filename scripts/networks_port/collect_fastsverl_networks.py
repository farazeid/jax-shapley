# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["torch==2.6.0", "numpy>=2.0"]
# ///
"""Capture SimpleNN outputs and gradients in a separate PyTorch environment.

Run with a checkout of the pinned source; never imported by normal pytest runs.
PPO examples exercise the configurable source builder, not paper PPO presets.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed git commands, no shell.
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Protocol, cast

import numpy as np

type FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float32]]


class Tensor(Protocol):
    """Typed boundary for the optional reference-only PyTorch installation."""

    grad: Tensor | None

    def detach(self) -> Tensor: ...
    def numpy(self) -> FloatArray: ...
    def requires_grad_(self, requires_grad: bool) -> Tensor: ...
    def __mul__(self, other: Tensor) -> Tensor: ...
    def sum(self) -> Tensor: ...
    def backward(self) -> None: ...


class TorchModule(Protocol):
    __version__: str

    def set_num_threads(self, threads: int) -> None: ...
    def manual_seed(self, seed: int) -> object: ...
    def from_numpy(self, array: FloatArray) -> Tensor: ...


REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"


class SourceOutput(Protocol):
    reference: Tensor


class SourceNetwork(Protocol):
    def __call__(self, inputs: Tensor) -> SourceOutput: ...
    def parameters(self) -> Iterator[Tensor]: ...


class SourceModule(Protocol):
    # Preserve the external source's constructor name at this adapter boundary.
    def SimpleNN(self, name: str, output_shape: tuple[int, ...], **kwargs: object) -> SourceNetwork: ...  # ruff: ignore[invalid-function-name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source_root = Path(args.source_root)
    output_path = Path(args.output)
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to verify fixture provenance")
    # Arguments are separate, with fixed git operations and no shell expansion.
    commit = subprocess.check_output([git, "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()  # ruff: ignore[subprocess-without-shell-equals-true]
    if commit != REFERENCE_COMMIT:
        raise ValueError(f"expected source commit {REFERENCE_COMMIT}, got {commit}")
    source_path = source_root / "fastsverl/fastsverl/models.py"
    committed = subprocess.check_output([git, "-C", str(source_root), "show", "HEAD:fastsverl/fastsverl/models.py"])  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed read-only git operation.
    if source_path.read_bytes() != committed:
        raise ValueError("source models.py has uncommitted changes")
    spec = importlib.util.spec_from_file_location("fastsverl_reference_models", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load upstream models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = cast(SourceModule, module)  # Narrow the dynamic source-module boundary.
    torch = cast(TorchModule, importlib.import_module("torch"))  # Optional reference-only library boundary.
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    arrays: dict[str, FloatArray] = {}
    for name, widths, ppo, actor in (
        ("dqn_gridworld", (2, 64, 64, 4), False, False),
        ("dqn_mastermind222", (8, 64, 64, 4), False, False),
        ("dqn_hypercube", (5, 128, 128, 10), False, False),
        ("dqn_mastermind333", (15, 128, 128, 27), False, False),
        ("ppo_actor", (8, 16, 4), True, True),
        ("ppo_critic", (8, 16, 1), True, False),
    ):
        arrays.update(capture(source, torch, name, widths, ppo=ppo, actor=actor))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, allow_pickle=False, **arrays)
    metadata = {
        "fastsverl_commit": commit,
        "source_sha256": hashlib.sha256(committed).hexdigest(),
        "fixture_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "seed": 1729,
        "dtype": "float32",
        "loss": "sum(outputs * probe)",
        "kernel_layout": "in_features, out_features (transposed from PyTorch)",
        "ppo_scope": "representative configurable SimpleNN architectures; no paper-specific PPO presets",
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def capture(
    source: SourceModule,
    torch: TorchModule,
    name: str,
    widths: tuple[int, ...],
    *,
    ppo: bool = False,
    actor: bool = False,
) -> dict[str, FloatArray]:
    architecture: dict[str, object] = {}
    for index, (input_width, output_width) in enumerate(pairwise(widths)):
        last = index == len(widths) - 2
        architecture["output" if last else f"linear{index}"] = ["Linear", [input_width, output_width]]
        if ppo:
            gain = (0.01 if actor else 1.0) if last else 2**0.5
            architecture[f"init{index}"] = ["layer_init", [gain, 0.0]]
        if not last:
            architecture[f"activation{index}"] = ["Tanh" if ppo else "ReLU", []]
        elif actor:
            architecture["softmax"] = ["Softmax", [-1]]
    model = source.SimpleNN("reference", (widths[-1],), **architecture)
    inputs = torch.from_numpy(np.arange(3 * widths[0], dtype=np.float32).reshape(3, widths[0]) % 5 - 1)
    inputs.requires_grad_(True)
    output = model(inputs).reference
    # A nonuniform upstream-output probe checks the full Jacobian contraction;
    # summing softmax alone would produce trivially zero actor gradients.
    output_array = output.detach().numpy()
    probe_array = np.linspace(-0.7, 0.9, output_array.size, dtype=np.float32).reshape(output_array.shape)
    probe = torch.from_numpy(probe_array)
    loss = (output * probe).sum()
    loss.backward()
    if inputs.grad is None:
        raise RuntimeError("missing reference input gradient")
    arrays = {
        f"{name}_inputs": inputs.detach().numpy(),
        f"{name}_outputs": output_array,
        f"{name}_probe": probe_array,
        f"{name}_loss": loss.detach().numpy(),
        f"{name}_input_grad": inputs.grad.numpy(),
    }
    parameters = list(model.parameters())
    for index, (weight, bias) in enumerate(zip(parameters[::2], parameters[1::2], strict=True)):
        if weight.grad is None or bias.grad is None:
            raise RuntimeError("missing reference parameter gradients")
        arrays[f"{name}_kernel{index}"] = weight.detach().numpy().T
        arrays[f"{name}_bias{index}"] = bias.detach().numpy()
        arrays[f"{name}_kernel_grad{index}"] = weight.grad.numpy().T
        arrays[f"{name}_bias_grad{index}"] = bias.grad.numpy()
    return arrays


if __name__ == "__main__":
    main()
