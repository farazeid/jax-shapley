"""Load the pinned explanation fixture into new_jax_shapley models; shared by parity tests."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from new_jax_shapley.algorithms import dqn
from new_jax_shapley.common.networks import MLP
from new_jax_shapley.envs import Gridworld, Hypercube, Mastermind

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_explanations.npz"
CASES = ("gridworld", "mastermind222", "hypercube")
KINDS = ("behaviour", "prediction")
ENVIRONMENTS: dict[str, dqn.Env] = {
    "gridworld": Gridworld(),
    "mastermind222": Mastermind(2, 2, 2),
    "hypercube": Hypercube(2, 5),
}
STEPS = 3
# Torch/JAX backend differences over three Adam steps, as in the DQN parity tests.
RTOL, ATOL = 3e-5, 3e-6

type FixtureArrays = dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class CaseLayout:
    name: str
    env: dqn.Env
    features: int
    actions: int
    width: int
    batch_size: int
    epsilon: float
    learning_rate: float


def load_fixture() -> FixtureArrays:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}


def layout(case: str) -> CaseLayout:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    shape = metadata["cases"][case]
    return CaseLayout(
        case,
        ENVIRONMENTS[case],
        int(shape["features"]),
        int(shape["actions"]),
        int(shape["width"]),
        int(metadata["batch_size"]),
        float(metadata["epsilon"]),
        float(metadata["learning_rate"]),
    )


def load_agent(fixture: FixtureArrays, case: CaseLayout) -> dqn.Agent:
    """The fixture's explained agent, with the fixture's explore epsilon."""
    config = dqn.DQNConfig(env=case.env, hidden_widths=(case.width,) * 2)
    model = dqn.initialize(config, jax.random.key(0)).model
    assign(model.mlp, fixture, f"{case.name}_agent")
    return dqn.Agent(model, jnp.asarray(case.epsilon, jnp.float32))


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
