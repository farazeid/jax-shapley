"""Controlled upstream fixtures establish learning fidelity, beyond compilation."""

from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.replay import TransitionBatch
from jax_shapley.train_dqn import (
    DQNTrainConfig,
    LossMetrics,
    action_log_probabilities,
    choose_action,
    dqn_loss,
    epsilon_at,
    exploration_probabilities,
    greedy_probabilities,
    initialize_models,
    td_targets,
    update,
    update_target,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_dqn.npz"


@pytest.mark.parametrize(
    "name,features,width,actions", [("gridworld", 2, 64, 4), ("mastermind222", 8, 64, 4), ("hypercube", 5, 128, 10)]
)
def test_source_loss_gradients_and_multiple_adam_steps(name: str, features: int, width: int, actions: int) -> None:
    training = DQNTrainConfig(gamma=0.9)
    model, target, optimizer, _ = initialize_models(
        dqn_config(features, actions, hidden_widths=(width, width)), training, jax.random.key(0)
    )
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for index, (source, destination) in enumerate(zip(model.mlp.layers, target.mlp.layers, strict=True)):
            source.kernel[...] = jnp.asarray(fixture[f"{name}_initial_{index * 2}"].T)
            source.bias[...] = jnp.asarray(fixture[f"{name}_initial_{index * 2 + 1}"])
            destination.kernel[...] = source.kernel[...].copy()
            destination.bias[...] = source.bias[...].copy()
        batch = TransitionBatch(
            jnp.asarray(fixture[f"{name}_observations"], jnp.int32),
            jnp.asarray(fixture[f"{name}_actions"], jnp.int32),
            jnp.asarray(fixture[f"{name}_rewards"]),
            jnp.asarray(fixture[f"{name}_next_observations"], jnp.int32),
            jnp.asarray(fixture[f"{name}_terminated"], jnp.bool_),
            jnp.zeros(4, jnp.float32),
        )
        np.testing.assert_allclose(
            greedy_probabilities(model(batch.observations).q_values), fixture[f"{name}_greedy"], atol=1e-7
        )
        np.testing.assert_allclose(
            exploration_probabilities(model(batch.observations).q_values, jnp.asarray(0.2)),
            fixture[f"{name}_explore"],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            td_targets(target, batch, training.gamma), fixture[f"{name}_targets"], rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            action_log_probabilities(greedy_probabilities(model(batch.observations).q_values), batch.actions),
            fixture[f"{name}_logprob"],
            rtol=1e-6,
            atol=1e-6,
        )
        for step in range(3):
            result = cast(
                tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(dqn_loss, has_aux=True)(model, target, batch, training.gamma),
            )
            (loss, _), gradients = result
            np.testing.assert_allclose(loss, fixture[f"{name}_loss_{step}"], rtol=3e-5, atol=3e-6)
            for index in range(len(model.mlp.layers)):
                layer = gradients["mlp"]["layers"][index]
                np.testing.assert_allclose(
                    cast(nnx.Param[Array], layer["kernel"])[...],
                    fixture[f"{name}_gradient_{step}_{index * 2}"].T,
                    rtol=3e-5,
                    atol=3e-6,
                )
                np.testing.assert_allclose(
                    cast(nnx.Param[Array], layer["bias"])[...],
                    fixture[f"{name}_gradient_{step}_{index * 2 + 1}"],
                    rtol=3e-5,
                    atol=3e-6,
                )
            metrics = update(model, target, optimizer, batch, training.gamma)
            assert jnp.isfinite(metrics.loss)
            update_target(model, target, training.tau)
            for index, (source, destination) in enumerate(zip(model.mlp.layers, target.mlp.layers, strict=True)):
                for label, layer in (("updated", source), ("target", destination)):
                    np.testing.assert_allclose(
                        layer.kernel[...], fixture[f"{name}_{label}_{step}_{index * 2}"].T, rtol=3e-5, atol=3e-6
                    )
                    np.testing.assert_allclose(
                        layer.bias[...], fixture[f"{name}_{label}_{step}_{index * 2 + 1}"], rtol=3e-5, atol=3e-6
                    )
        for step in (0, 25, 50, 100):
            config = DQNTrainConfig(total_timesteps=100)
            np.testing.assert_allclose(
                epsilon_at(config, jnp.asarray(step, jnp.int32)), fixture[f"epsilon_{step}"], atol=1e-7
            )


def test_ties_categorical_logprobs_and_explicit_key() -> None:
    values = jnp.array([[2, 2, 0, -1], [1, 1, 1, 1]], jnp.float32)
    greedy = greedy_probabilities(values)
    np.testing.assert_array_equal(greedy, [[0.5, 0.5, 0, 0], [0.25, 0.25, 0.25, 0.25]])
    np.testing.assert_allclose(
        action_log_probabilities(greedy, jnp.array([2, 1], jnp.int32)),
        [np.log(np.finfo(np.float32).eps), np.log(0.25)],
        rtol=1e-6,
    )
    model = QNetwork(dqn_config(2, 4, hidden_widths=()), rngs=nnx.Rngs(0))
    model.mlp.layers[0].kernel[...] = jnp.zeros((2, 4), jnp.float32)
    model.mlp.layers[0].bias[...] = values[0]
    key = jax.random.key(5)
    result, next_key = choose_action(model, jnp.zeros((128, 2), jnp.int32), jnp.asarray(0, jnp.float32), key)
    assert set(np.asarray(result.actions).tolist()) == {0, 1}
    assert result.actions.dtype == jnp.int32
    assert result.log_probabilities.dtype == jnp.float32
    assert not jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))


def test_target_initialization_independence_and_termination_mask() -> None:
    from tests.test_replay import batch

    model, target, _, _ = initialize_models(dqn_config(2, 4), DQNTrainConfig(), jax.random.key(0))
    np.testing.assert_array_equal(model.mlp.layers[0].kernel[...], target.mlp.layers[0].kernel[...])
    model.mlp.layers[0].kernel[...] += 1
    assert not jnp.array_equal(model.mlp.layers[0].kernel[...], target.mlp.layers[0].kernel[...])
    data = batch(0, 4)
    targets = td_targets(target, data, 1.0)
    np.testing.assert_array_equal(targets[jnp.array([0, 2])], data.rewards[jnp.array([0, 2])])
