"""Whole explanation runs traced, lowered and compiled at production buffer and network shapes."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_shapley.characteristics import CharacteristicConfig, CharacteristicModel, Kind, characteristic_network
from jax_shapley.envs import Gridworld, Hypercube, Mastermind222
from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.train_explanations import (
    Buffers,
    CharacteristicRunConfig,
    ExplanationTrainConfig,
    FrozenAgent,
    ShapleyRunConfig,
    run_characteristic,
    run_shapley,
)
from jax_shapley.wrappers import EnvironmentProtocol
from tests.explanation_support import RecordingSink

KEY = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)
# Source settings: Gridworld/Mastermind defaults; Hypercube's separate exact buffer and validation fraction.
DEFAULT = ExplanationTrainConfig()
HYPERCUBE = ExplanationTrainConfig(
    epochs=10_000,
    val_fraction=16 * 10 / 10_000,
    exact_buffer_size=100_000,
    exact_condition=0.01,
    track_batch=True,
)
CASES = ("gridworld", "mastermind222", "hypercube")


@pytest.mark.compile
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["behaviour", "prediction"])
def test_whole_explanation_runs_compile(case: str, kind: Kind) -> None:
    if case == "gridworld":
        _compile(Gridworld, 2, 4, 64, DEFAULT, 8, kind)
    elif case == "mastermind222":
        _compile(Mastermind222, 8, 4, 64, DEFAULT, 8000, kind)
    else:
        _compile(lambda: Hypercube(6, 5), 5, 10, 128, HYPERCUBE, 7776, kind)


def _compile[State, Params](
    factory: Callable[[], EnvironmentProtocol[State, Params]],
    features: int,
    actions: int,
    width: int,
    training: ExplanationTrainConfig,
    max_unique: int,
    kind: Kind,
) -> None:
    explained = CharacteristicConfig(kind, hidden_widths=(width, width))
    agent = FrozenAgent(
        QNetwork(dqn_config(features, actions, hidden_widths=(width, width)), rngs=nnx.Rngs(0)),
        jnp.asarray(0.05, jnp.float32),
    )
    characteristic = CharacteristicModel(characteristic_network(features, actions, explained), rngs=nnx.Rngs(1))
    char_config = CharacteristicRunConfig(agent=_unused(), kind=kind, hidden_widths=(width, width), train=training)
    run = partial(
        run_characteristic, environment_factory=factory, config=char_config, max_unique=max_unique, sink=RecordingSink()
    )
    nnx.jit(run).trace(KEY, agent, None, None, None).lower().compile()
    inherited = Buffers(
        jnp.zeros((training.train_size, features), jnp.int32),
        jnp.zeros((training.validation_size, features), jnp.int32),
        jnp.zeros((training.exact_size, features), jnp.int32),
    )
    for buffers, with_exact_char in (("fresh", True), ("characteristic", False)):
        shapley_config = ShapleyRunConfig(
            agent=_unused(),
            characteristic=_unused(),
            with_exact_char=with_exact_char,
            buffers=buffers,
            hidden_widths=(width, width),
            train=training,
        )
        shapley = partial(
            run_shapley,
            environment_factory=factory,
            config=shapley_config,
            explained=explained,
            max_unique=max_unique,
            sink=RecordingSink(),
        )
        nnx.jit(shapley).trace(KEY, agent, characteristic, inherited, None).lower().compile()


def _unused() -> Path:
    """Run configs require artifact paths; compile tests pass the models explicitly."""
    return Path("unused")
