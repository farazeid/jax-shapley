"""Whole explanation runs traced, lowered and compiled at production buffer and network shapes."""

from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    Estimator,
    Kind,
    characteristic_network,
)
from jax_shapley.envs import Gridworld, Hypercube, Mastermind, Mastermind222
from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.outcome import OutcomeNetworks, Transitions, outcome_network
from jax_shapley.tabular import transition_model
from jax_shapley.train_dqn import EnvironmentConfig, GridworldConfig, Mastermind222Config
from jax_shapley.train_explanations import (
    Buffers,
    CharacteristicRunConfig,
    ExplainedOutcome,
    ExplanationTrainConfig,
    FrozenAgent,
    OutcomeInputs,
    OutcomeRunConfig,
    OutcomeTrainConfig,
    ShapleyRunConfig,
    run_characteristic,
    run_outcome,
    run_outcome_on_policy,
    run_outcome_shapley,
    run_shapley,
    stack_seeds,
)
from jax_shapley.wrappers import EnvironmentProtocol
from tests.explanation_support import RecordingSink

# A three-seed sweep: the production path for single runs (one seed) and sweeps alike.
SEEDS = 3
KEYS = jax.ShapeDtypeStruct((SEEDS,), jax.random.key(0).dtype)
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
SHAPLEY = ExplanationTrainConfig(epochs=20)


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


@pytest.mark.compile
@pytest.mark.parametrize("case", ["gridworld", "mastermind222"])
@pytest.mark.parametrize("with_exact_char", [True, False])
def test_whole_outcome_runs_compile(case: str, with_exact_char: bool) -> None:
    """Audited dqn_performance_* settings: replay prefixes, 128 epochs with track_batch, batch 64."""
    if case == "gridworld":
        training = OutcomeTrainConfig(train_size=50_000, validation_size=8 * 10)
        config = OutcomeRunConfig(_unused(), _unused(), with_exact_char=with_exact_char, tau=0.001, train=training)
        _compile_outcome(Gridworld, GridworldConfig(), 2, config)
    else:
        training = OutcomeTrainConfig(train_size=100_000, validation_size=16 * 10)
        config = OutcomeRunConfig(_unused(), _unused(), with_exact_char=with_exact_char, tau=0.01, train=training)
        _compile_outcome(Mastermind222, Mastermind222Config(), 8, config)


@pytest.mark.compile
@pytest.mark.parametrize("case", ["gridworld", "mastermind222"])
@pytest.mark.parametrize("estimator", ["model", "sampled"])
def test_whole_on_policy_outcome_runs_compile(case: str, estimator: Estimator) -> None:
    """Audited train_V_on_policy settings: Gridworld 100k steps over 8 environments, Mastermind 200k over 16.

    The sampling groups condition on a sampled behaviour (greedy, 16 environments; Mastermind 1M steps).
    """
    if case == "gridworld":
        training = OutcomeTrainConfig(total_timesteps=100_000, num_envs=8 if estimator == "model" else 16)
        config = OutcomeRunConfig(_unused(), _unused(), "on_policy", tau=0.001, train=training, chunk_epochs=125)
        _compile_on_policy(Gridworld, GridworldConfig(), 2, config, estimator)
    else:
        steps = 200_000 if estimator == "model" else 1_000_000
        training = OutcomeTrainConfig(total_timesteps=steps, num_envs=16)
        config = OutcomeRunConfig(_unused(), _unused(), "on_policy", tau=0.01, train=training, chunk_epochs=125)
        _compile_on_policy(Mastermind222, Mastermind222Config(), 8, config, estimator)


@pytest.mark.compile
@pytest.mark.parametrize("case", ["gridworld", "mastermind222", "mastermind333"])
@pytest.mark.parametrize("kind", ["behaviour", "prediction"])
def test_whole_sampled_runs_compile(case: str, kind: Kind) -> None:
    """sampling/shapleys settings: the pool is the 8,000-row train buffer, which is also the exact buffer."""
    if case == "gridworld":
        _compile_sampled(Gridworld, 2, 4, 64, 8, kind)
    elif case == "mastermind222":
        _compile_sampled(Mastermind222, 8, 4, 64, 8000, kind)
    else:
        _compile_sampled(lambda: Mastermind(3, 3, 3), 15, 27, 128, 4096, kind)


def _compile_sampled[State, Params](
    factory: Callable[[], EnvironmentProtocol[State, Params]],
    features: int,
    actions: int,
    width: int,
    max_unique: int,
    kind: Kind,
) -> None:
    explained = CharacteristicConfig(kind, hidden_widths=(width, width), estimator="sampled")
    agent = FrozenAgent(
        QNetwork(dqn_config(features, actions, hidden_widths=(width, width)), rngs=nnx.Rngs(0)),
        jnp.asarray(0.05, jnp.float32),
    )
    agents = stack_seeds([agent] * SEEDS)
    collect = ExplanationTrainConfig(exact_loss=False)
    char_config = CharacteristicRunConfig(
        agent=_unused(), kind=kind, hidden_widths=(width, width), estimator="sampled", train=collect
    )
    run = partial(
        run_characteristic, environment_factory=factory, config=char_config, max_unique=max_unique, sink=RecordingSink()
    )
    nnx.jit(run).trace(KEYS, agents, None, None, None).lower().compile()
    training = ExplanationTrainConfig(epochs=300)
    pool = jnp.zeros((training.train_size, features), jnp.int32)
    inherited = Buffers(pool, jnp.zeros((training.validation_size, features), jnp.int32), pool)
    shapley_config = ShapleyRunConfig(
        agent=_unused(),
        characteristic=_unused(),
        buffers="characteristic",
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
    characteristic = CharacteristicModel(characteristic_network(features, actions, explained), rngs=nnx.Rngs(1))
    characteristics, buffers = stack_seeds([characteristic] * SEEDS), stack_seeds([inherited] * SEEDS)
    nnx.jit(shapley).trace(KEYS, agents, characteristics, buffers, None).lower().compile()


def _compile_on_policy[State, Params](
    factory: Callable[[], EnvironmentProtocol[State, Params]],
    environment: EnvironmentConfig,
    features: int,
    config: OutcomeRunConfig,
    estimator: Estimator = "model",
) -> None:
    actions, width = 4, 64
    policy = "explore" if estimator == "model" else "greedy"
    explained = CharacteristicConfig("behaviour", policy, (width, width), estimator)
    agent = FrozenAgent(
        QNetwork(dqn_config(features, actions, hidden_widths=(width, width)), rngs=nnx.Rngs(0)),
        jnp.asarray(0.05, jnp.float32),
    )
    characteristic = CharacteristicModel(characteristic_network(features, actions, explained), rngs=nnx.Rngs(1))
    buffers = Buffers(
        jnp.zeros((DEFAULT.train_size, features), jnp.int32),
        jnp.zeros((DEFAULT.validation_size, features), jnp.int32),
        jnp.zeros((DEFAULT.exact_size, features), jnp.int32),
    )
    run = partial(
        run_outcome_on_policy,
        environment_factory=factory,
        config=config,
        explained=explained,
        tabular=transition_model(environment),
        sink=RecordingSink(),
    )
    arguments = (stack_seeds([agent] * SEEDS), stack_seeds([characteristic] * SEEDS), stack_seeds([buffers] * SEEDS))
    nnx.jit(run).trace(KEYS, *arguments, None).lower().compile()


def _compile_outcome[State, Params](
    factory: Callable[[], EnvironmentProtocol[State, Params]],
    environment: EnvironmentConfig,
    features: int,
    config: OutcomeRunConfig,
) -> None:
    actions, width = 4, 64
    explained = CharacteristicConfig("behaviour", "explore", (width, width))
    agent = FrozenAgent(
        QNetwork(dqn_config(features, actions, hidden_widths=(width, width)), rngs=nnx.Rngs(0)),
        jnp.asarray(0.05, jnp.float32),
    )
    characteristic = CharacteristicModel(characteristic_network(features, actions, explained), rngs=nnx.Rngs(1))
    behaviour_buffers = Buffers(
        jnp.zeros((DEFAULT.train_size, features), jnp.int32),
        jnp.zeros((DEFAULT.validation_size, features), jnp.int32),
        jnp.zeros((DEFAULT.exact_size, features), jnp.int32),
    )
    training = config.train
    inputs = OutcomeInputs(
        behaviour_buffers,
        _transitions(training.train_size, features),
        _transitions(training.validation_size, features),
    )
    run = partial(
        run_outcome,
        environment_factory=factory,
        config=config,
        explained=explained,
        tabular=transition_model(environment),
        sink=RecordingSink(),
    )
    arguments = (stack_seeds([agent] * SEEDS), stack_seeds([characteristic] * SEEDS), stack_seeds([inputs] * SEEDS))
    nnx.jit(run).trace(KEYS, *arguments, None).lower().compile()
    # Source PerformanceShapley settings (Gridworld's 20 epochs; Mastermind's 100 share the shapes).
    explained_outcome = ExplainedOutcome(
        OutcomeNetworks(outcome_network(features, actions, config.outcome), rngs=nnx.Rngs(2)), characteristic
    )
    shapley = partial(
        run_outcome_shapley,
        environment_factory=factory,
        config=ShapleyRunConfig(_unused(), _unused(), with_exact_char=config.with_exact_char, train=SHAPLEY),
        explained=explained,
        outcome=config,
        tabular=transition_model(environment),
        sink=RecordingSink(),
    )
    frozen = (stack_seeds([agent] * SEEDS), stack_seeds([explained_outcome] * SEEDS))
    nnx.jit(shapley).trace(KEYS, *frozen, stack_seeds([behaviour_buffers] * SEEDS), None).lower().compile()


def _transitions(rows: int, features: int) -> Transitions:
    observations = jnp.zeros((rows, features), jnp.int32)
    return Transitions(
        observations,
        jnp.zeros((rows,), jnp.int32),
        jnp.zeros((rows,), jnp.float32),
        observations,
        jnp.zeros((rows,), jnp.bool_),
    )


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
    agents = stack_seeds([agent] * SEEDS)
    nnx.jit(run).trace(KEYS, agents, None, None, None).lower().compile()
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
        characteristics, buffers = stack_seeds([characteristic] * SEEDS), stack_seeds([inherited] * SEEDS)
        nnx.jit(shapley).trace(KEYS, agents, characteristics, buffers, None).lower().compile()


def _unused() -> Path:
    """Run configs require artifact paths; compile tests pass the models explicitly."""
    return Path("unused")
