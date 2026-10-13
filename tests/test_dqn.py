"""Controlled upstream fixtures establish learning fidelity, beyond compilation."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.networks import soft_update
from jax_shapley.envs import Gridworld, Hypercube, Mastermind

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_dqn.npz"
ENVIRONMENTS = {"gridworld": Gridworld(), "mastermind222": Mastermind(2, 2, 2), "hypercube": Hypercube(2, 5)}


@pytest.mark.parametrize("name,width", [("gridworld", 64), ("mastermind222", 64), ("hypercube", 128)])
def test_source_loss_gradients_and_multiple_adam_steps(name: str, width: int) -> None:
    env = ENVIRONMENTS[name]
    config = dqn.DQNConfig(env=env, hidden_widths=(width, width), gamma=0.9)
    state = dqn.initialize(config, jax.random.key(0))
    model, target, optimizer = state.model, state.target, state.optimizer
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        for index, (source, destination) in enumerate(zip(model.mlp.layers, target.mlp.layers, strict=True)):
            source.kernel[...] = jnp.asarray(fixture[f"{name}_initial_{index * 2}"].T)
            source.bias[...] = jnp.asarray(fixture[f"{name}_initial_{index * 2 + 1}"])
            destination.kernel[...] = source.kernel[...].copy()
            destination.bias[...] = source.bias[...].copy()
        batch = dqn.TransitionBatch(
            jnp.asarray(fixture[f"{name}_observations"], jnp.int32),
            jnp.asarray(fixture[f"{name}_actions"], jnp.int32),
            jnp.asarray(fixture[f"{name}_rewards"]),
            jnp.asarray(fixture[f"{name}_next_observations"], jnp.int32),
            jnp.asarray(fixture[f"{name}_terminated"], jnp.bool_),
            jnp.zeros(4, jnp.float32),
        )
        greedy = behaviour.targets(
            CharacteristicConfig(env, policy="greedy"), dqn.Agent(model, jnp.asarray(0.0)), batch.observations
        )
        np.testing.assert_allclose(greedy, fixture[f"{name}_greedy"], atol=1e-7)
        explore = behaviour.targets(
            CharacteristicConfig(env, policy="explore"), dqn.Agent(model, jnp.asarray(0.2)), batch.observations
        )
        np.testing.assert_allclose(explore, fixture[f"{name}_explore"], atol=1e-7)
        np.testing.assert_allclose(
            dqn.log_probabilities(greedy, batch.actions), fixture[f"{name}_logprob"], rtol=1e-6, atol=1e-6
        )
        for step in range(3):
            loss, gradients = cast(
                tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(dqn.dqn_loss)(model, target, batch, config.gamma),
            )
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
            # The DQN step's learning update, then its soft target update.
            optimizer.update(model, gradients)
            soft_update(model.mlp, target.mlp, config.tau)
            for index, (source, destination) in enumerate(zip(model.mlp.layers, target.mlp.layers, strict=True)):
                for label, layer in (("updated", source), ("target", destination)):
                    np.testing.assert_allclose(
                        layer.kernel[...], fixture[f"{name}_{label}_{step}_{index * 2}"].T, rtol=3e-5, atol=3e-6
                    )
                    np.testing.assert_allclose(
                        layer.bias[...], fixture[f"{name}_{label}_{step}_{index * 2 + 1}"], rtol=3e-5, atol=3e-6
                    )
        # The explained agent's epsilon is the schedule's at the last step taken.
        schedule = dqn.DQNConfig(env=env, total_timesteps=100)
        for step in (0, 25, 50, 100):
            stepped = replace(state, vector_step=jnp.asarray(step + 1, jnp.int32))
            np.testing.assert_allclose(
                dqn.trained_agent(schedule, stepped).epsilon, fixture[f"epsilon_{step}"], atol=1e-7
            )


def test_ties_categorical_logprobs_and_explicit_key() -> None:
    values = jnp.array([[2, 2, 0, -1], [1, 1, 1, 1]], jnp.float32)
    env = Gridworld()
    config = dqn.DQNConfig(env=env, hidden_widths=())
    model = dqn.initialize(config, jax.random.key(0)).model
    model.mlp.layers[0].kernel[...] = jnp.zeros((2, 4), jnp.float32)
    model.mlp.layers[0].bias[...] = values[1]
    greedy = behaviour.targets(
        CharacteristicConfig(env), dqn.Agent(model, jnp.asarray(0.0)), jnp.zeros((1, 2), jnp.int32)
    )
    np.testing.assert_array_equal(greedy, [[0.25, 0.25, 0.25, 0.25]])
    model.mlp.layers[0].bias[...] = values[0]
    greedy = behaviour.targets(
        CharacteristicConfig(env), dqn.Agent(model, jnp.asarray(0.0)), jnp.zeros((1, 2), jnp.int32)
    )
    np.testing.assert_array_equal(greedy, [[0.5, 0.5, 0, 0]])
    np.testing.assert_allclose(
        dqn.log_probabilities(jnp.concatenate([greedy, jnp.full((1, 4), 0.25)]), jnp.array([2, 1], jnp.int32)),
        [np.log(np.finfo(np.float32).eps), np.log(0.25)],
        rtol=1e-6,
    )
    key = jax.random.key(5)
    actions, log_probabilities, next_key = dqn.choose_action(
        model, jnp.zeros((128, 2), jnp.int32), jnp.asarray(0, jnp.float32), key
    )
    assert set(np.asarray(actions).tolist()) == {0, 1}
    assert actions.dtype == jnp.int32
    assert log_probabilities.dtype == jnp.float32
    assert not jnp.array_equal(jax.random.key_data(next_key), jax.random.key_data(key))


def test_target_initialization_independence_and_termination_mask() -> None:
    config = dqn.DQNConfig(env=Gridworld(), gamma=1.0)
    state = dqn.initialize(config, jax.random.key(0))
    model, target = state.model, state.target
    np.testing.assert_array_equal(model.mlp.layers[0].kernel[...], target.mlp.layers[0].kernel[...])
    model.mlp.layers[0].kernel[...] += 1
    assert not jnp.array_equal(model.mlp.layers[0].kernel[...], target.mlp.layers[0].kernel[...])
    observations = jnp.asarray([[0, 0], [1, 1], [2, 2], [3, 0]], jnp.int32)
    actions = jnp.asarray([0, 1, 2, 3], jnp.int32)
    rewards = jnp.asarray([1.0, -1.0, 0.5, 2.0], jnp.float32)
    batch = dqn.TransitionBatch(observations, actions, rewards, observations, jnp.ones(4, jnp.bool_), rewards)
    # Terminated rows bootstrap nothing: the loss is against the rewards alone.
    values = jnp.take_along_axis(model(observations).q_values, actions[:, None], axis=-1)[:, 0]
    np.testing.assert_allclose(dqn.dqn_loss(model, target, batch, 1.0), jnp.mean(jnp.square(rewards - values)))
