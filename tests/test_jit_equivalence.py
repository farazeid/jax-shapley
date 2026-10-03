"""Compiled explanation numerics must equal eager evaluation at production batch sizes.

XLA CPU (jax 0.11.0) miscompiles networks applied to broadcast-constant batches of
64 or more rows. Fixture parity runs eagerly on small batches, so these checks
guard the compiled paths the training loop actually uses.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    characteristic_loss,
    characteristic_network,
    exact_characteristic,
    exact_characteristic_loss,
)
from jax_shapley.coalitions import all_coalitions, kernel_coalitions, uniform_coalitions
from jax_shapley.envs import Mastermind222
from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.shapley import (
    MaskedCharacteristic,
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    explain,
    shapley_loss,
    shapley_network,
)
from jax_shapley.train_explanations import (
    ExplanationTrainConfig,
    FrozenAgent,
    ShapleyRunConfig,
    collect_buffers,
    run_shapley,
    seed_slice,
    stack_seeds,
)
from jax_shapley.wrappers import EpisodeStatisticsWrapper
from tests.explanation_support import RecordingSink

RTOL, ATOL = 1e-5, 1e-6


@pytest.mark.parametrize("rows", [63, 64, 128, 2000])
@pytest.mark.parametrize("features,actions,width", [(2, 4, 64), (5, 10, 32)], ids=["gridworld", "hypercube"])
def test_shapley_targets_and_values_match_eager(rows: int, features: int, actions: int, width: int) -> None:
    characteristic, model = _models(features=features, actions=actions, width=width)
    observations = jax.random.randint(jax.random.key(1), (rows, features), 0, 2, jnp.int32)
    coalition = kernel_coalitions(jax.random.key(2), 1, features)[0]
    function = MaskedCharacteristic(characteristic)
    np.testing.assert_allclose(
        nnx.jit(lambda c, o: MaskedCharacteristic(c).null(o))(characteristic, observations),
        function.null(observations),
        rtol=RTOL,
        atol=ATOL,
    )
    compiled = nnx.jit(lambda m, c, o, k: shapley_loss(m, MaskedCharacteristic(c), o, k)[0])(
        model, characteristic, observations, coalition
    )
    np.testing.assert_allclose(
        compiled, shapley_loss(model, function, observations, coalition)[0], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(
        nnx.jit(lambda m, c, o: explain(m, MaskedCharacteristic(c), o))(model, characteristic, observations),
        explain(model, function, observations),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("rows", [64, 2000])
def test_characteristic_losses_and_tables_match_eager(rows: int) -> None:
    characteristic, model = _models(features=5, actions=10, width=32)
    observations = jax.random.randint(jax.random.key(3), (rows, 5), 0, 3, jnp.int32)
    targets = jax.nn.softmax(jax.random.normal(jax.random.key(4), (rows, 10)))
    coalitions = uniform_coalitions(jax.random.key(5), rows, 5).at[0].set(False)
    np.testing.assert_allclose(
        nnx.jit(lambda c, o, t, k: characteristic_loss(c, o, t, k)[0])(
            characteristic, observations, targets, coalitions
        ),
        characteristic_loss(characteristic, observations, targets, coalitions)[0],
        rtol=RTOL,
        atol=ATOL,
    )
    every = all_coalitions(5)
    exact = jax.jit(partial(exact_characteristic, coalitions=every, max_unique=243))(observations, targets)
    np.testing.assert_allclose(
        nnx.jit(exact_characteristic_loss)(characteristic, exact),
        exact_characteristic_loss(characteristic, exact),
        rtol=RTOL,
    )
    function = MaskedCharacteristic(characteristic)
    table = nnx.jit(lambda c, s, k: characteristic_table(MaskedCharacteristic(c), s, k))(
        characteristic, exact.states, every
    )
    np.testing.assert_allclose(table, characteristic_table(function, exact.states, every), rtol=RTOL, atol=ATOL)
    solved = exact_shapley(table)
    np.testing.assert_allclose(
        nnx.jit(lambda m, c, s, v, x: exact_shapley_loss(m, MaskedCharacteristic(c), s, v, x))(
            model, characteristic, exact.states, exact.state_valid, solved
        ),
        exact_shapley_loss(model, function, exact.states, exact.state_valid, solved),
        rtol=RTOL,
    )


def test_logged_exact_loss_matches_eager_evaluation_of_the_returned_model() -> None:
    """End to end: the best epoch's logged exact loss is the restored best model's eager exact loss."""
    training = ExplanationTrainConfig(epochs=6, batch_size=128, buffer_size=640, num_envs=16)
    agent = FrozenAgent(QNetwork(dqn_config(8, 4, hidden_widths=(16,)), rngs=nnx.Rngs(0)), jnp.asarray(0.1))
    characteristic = CharacteristicModel(
        characteristic_network(8, 4, CharacteristicConfig("behaviour", hidden_widths=(16,))), rngs=nnx.Rngs(2)
    )
    env = EpisodeStatisticsWrapper(Mastermind222())
    _, inherited = collect_buffers(env, agent, "greedy", training, jax.random.key(1), None)
    config = ShapleyRunConfig(
        agent=Path("unused"), characteristic=Path("unused"), with_exact_char=False, hidden_widths=(16,), train=training
    )
    sink = RecordingSink()
    runner = nnx.jit(
        partial(
            run_shapley,
            environment_factory=Mastermind222,
            config=config,
            explained=CharacteristicConfig("behaviour"),
            max_unique=training.exact_size,
            sink=sink,
        )
    )
    batched = runner(
        jnp.stack([jax.random.key(0)]),
        stack_seeds([agent]),
        stack_seeds([characteristic]),
        stack_seeds([inherited]),
        None,
    )
    result = seed_slice(batched, 0)
    exact = exact_characteristic(
        inherited.exact, jnp.zeros((training.exact_size, 4)), all_coalitions(8), max_unique=training.exact_size
    )
    function = MaskedCharacteristic(characteristic)
    solved = exact_shapley(characteristic_table(function, exact.states, all_coalitions(8)))
    eager = exact_shapley_loss(result.model, function, exact.states, exact.state_valid, solved)
    logged = {row[0]: row[4] for row in sink.rows}
    np.testing.assert_allclose(logged[int(result.best_epoch)], eager, rtol=1e-5)


def _models(*, features: int, actions: int, width: int) -> tuple[CharacteristicModel, ShapleyModel]:
    config = CharacteristicConfig("behaviour", hidden_widths=(width, width))
    characteristic = CharacteristicModel(characteristic_network(features, actions, config), rngs=nnx.Rngs(0))
    layout = ShapleyLayout(features, actions)
    return characteristic, ShapleyModel(shapley_network(layout, (width, width)), layout, rngs=nnx.Rngs(1))
