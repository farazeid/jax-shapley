"""Load the pinned explanation fixture into NNX models; shared by parity tests."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from jax_shapley.explanation_io import ExplanationSnapshot
from jax_shapley.networks import MLP, QNetwork, dqn_config
from jax_shapley.run_io import CallbackToken

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_explanations.npz"
CASES = ("gridworld", "mastermind222", "hypercube")
KINDS = ("behaviour", "prediction")
STEPS = 3
# Torch/JAX backend differences over three Adam steps, as in the DQN parity tests.
RTOL, ATOL = 3e-5, 3e-6

type FixtureArrays = dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class CaseLayout:
    name: str
    features: int
    actions: int
    width: int
    batch_size: int
    epsilon: float
    learning_rate: float


@dataclass
class RecordingSink:
    """Inert explanation sink: records epoch rows and checkpoint (epoch, phase) pairs."""

    rows: list[tuple[int, int, float, float, float, int]] = field(default_factory=list)
    checkpoints: list[tuple[int, int]] = field(default_factory=list)

    def epochs(self, metrics: tuple[Array, ...]) -> CallbackToken:
        epoch, step, train_loss, validation_loss, exact_loss, best_epoch, active = (np.asarray(m) for m in metrics)
        for index in np.flatnonzero(active):
            self.rows.append((
                int(epoch[index]),
                int(step[index]),
                float(train_loss[index]),
                float(validation_loss[index]),
                float(exact_loss[index]),
                int(best_epoch[index]),
            ))
        return np.asarray(0, np.int32)

    def checkpoint(self, snapshot: ExplanationSnapshot) -> CallbackToken:
        self.checkpoints.append((int(snapshot.epoch), int(snapshot.phase)))
        return np.asarray(0, np.int32)


def load_fixture() -> FixtureArrays:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}


def layout(case: str) -> CaseLayout:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    shape = metadata["cases"][case]
    return CaseLayout(
        case,
        int(shape["features"]),
        int(shape["actions"]),
        int(shape["width"]),
        int(metadata["batch_size"]),
        float(metadata["epsilon"]),
        float(metadata["learning_rate"]),
    )


def load_agent(fixture: FixtureArrays, case: CaseLayout) -> QNetwork:
    agent = QNetwork(dqn_config(case.features, case.actions, hidden_widths=(case.width,) * 2), rngs=nnx.Rngs(0))
    assign(agent.mlp, fixture, f"{case.name}_agent")
    return agent


def assign(mlp: MLP, fixture: FixtureArrays, prefix: str) -> None:
    """Source parameters alternate weight (out, in) and bias per Linear layer."""
    for index, layer in enumerate(mlp.layers):
        layer.kernel[...] = jnp.asarray(fixture[f"{prefix}_{index * 2}"].T)
        layer.bias[...] = jnp.asarray(fixture[f"{prefix}_{index * 2 + 1}"])


def assert_parameters(mlp: MLP, fixture: FixtureArrays, prefix: str, *, rtol: float = RTOL, atol: float = ATOL) -> None:
    for index, layer in enumerate(mlp.layers):
        np.testing.assert_allclose(layer.kernel[...], fixture[f"{prefix}_{index * 2}"].T, rtol=rtol, atol=atol)
        np.testing.assert_allclose(layer.bias[...], fixture[f"{prefix}_{index * 2 + 1}"], rtol=rtol, atol=atol)


def assert_gradients(gradients: nnx.State[str | int, nnx.Param[Array]], fixture: FixtureArrays, prefix: str) -> None:
    layers = gradients["mlp"]["layers"]
    for index in range(3):
        layer = layers[index]
        np.testing.assert_allclose(
            cast(nnx.Param[Array], layer["kernel"])[...], fixture[f"{prefix}_{index * 2}"].T, rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            cast(nnx.Param[Array], layer["bias"])[...], fixture[f"{prefix}_{index * 2 + 1}"], rtol=RTOL, atol=ATOL
        )
