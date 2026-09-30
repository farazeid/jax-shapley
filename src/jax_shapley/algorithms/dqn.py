"""FastSVERL DQN policies and differentiable learning units."""

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from jax import Array

from jax_shapley.config import DQNTrainConfig
from jax_shapley.networks import MLPConfig, ModelInput, QNetwork
from jax_shapley.replay import TransitionBatch


class ActionOutput(NamedTuple):
    actions: Array
    log_probabilities: Array


@struct.dataclass
class LossMetrics:
    loss: Array
    target_mean: Array
    value_mean: Array


def greedy_probabilities(q_values: Array) -> Array:
    ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
    return ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)


def exploration_probabilities(q_values: Array, epsilon: Array) -> Array:
    return epsilon / q_values.shape[-1] + (1 - epsilon) * greedy_probabilities(q_values)


def epsilon_at(config: DQNTrainConfig, global_step: Array) -> Array:
    fraction = global_step.astype(jnp.float32) / (config.exploration_fraction * config.total_timesteps)
    return jnp.maximum(config.start_e + (config.end_e - config.start_e) * fraction, config.end_e).astype(jnp.float32)


def action_log_probabilities(probabilities: Array, actions: Array) -> Array:
    # Torch Categorical renormalizes probabilities and clamps before converting to logits.
    probabilities = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    epsilon = jnp.finfo(jnp.float32).eps
    logits = jnp.log(jnp.clip(probabilities, epsilon, 1 - epsilon))
    return jnp.take_along_axis(logits, actions[..., None], axis=-1)[..., 0]


def choose_action(model: QNetwork, observations: ModelInput, epsilon: Array, key: Array) -> tuple[ActionOutput, Array]:
    probabilities = exploration_probabilities(model(observations).q_values, epsilon)
    keys: Array = jax.random.split(key)
    # Sampling uses true zero-probability logits; the clamp applies only to reported log probabilities.
    actions: Array = jax.random.categorical(keys[1], jnp.log(probabilities), axis=-1).astype(jnp.int32)
    return ActionOutput(actions, action_log_probabilities(probabilities, actions)), keys[0]


def td_targets(target: QNetwork, batch: TransitionBatch, gamma: float) -> Array:
    values = jnp.max(target(batch.next_observations).q_values, axis=-1)
    return jax.lax.stop_gradient(batch.rewards + gamma * values * (~batch.terminated).astype(jnp.float32))


def dqn_loss(model: QNetwork, target: QNetwork, batch: TransitionBatch, gamma: float) -> tuple[Array, LossMetrics]:
    batch.validate()
    targets = td_targets(target, batch, gamma)
    values = jnp.take_along_axis(model(batch.observations).q_values, batch.actions[:, None], axis=-1)[:, 0]
    loss = jnp.mean(jnp.square(targets - values))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(values))


def make_optimizer(model: QNetwork, config: DQNTrainConfig) -> nnx.Optimizer[QNetwork]:
    adam = config.adam
    return nnx.Optimizer(
        model,
        optax.adam(config.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
        wrt=nnx.Param,
    )


def initialize_models(
    config: MLPConfig, training: DQNTrainConfig, key: Array
) -> tuple[QNetwork, QNetwork, nnx.Optimizer[QNetwork], Array]:
    keys: Array = jax.random.split(key)
    model = QNetwork(config, rngs=nnx.Rngs(keys[1]))
    target = nnx.clone(model)
    return model, target, make_optimizer(model, training), keys[0]


def update(
    model: QNetwork, target: QNetwork, optimizer: nnx.Optimizer[QNetwork], batch: TransitionBatch, gamma: float
) -> LossMetrics:
    # NNX transformation has a dynamic return signature; assert its known loss/state boundary.
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(dqn_loss, has_aux=True)(model, target, batch, gamma),
    )
    (_, metrics), gradients = result
    optimizer.update(model, gradients)
    return metrics


def update_target(model: QNetwork, target: QNetwork, tau: float) -> None:
    for source, destination in zip(model.mlp.layers, target.mlp.layers, strict=True):
        destination.kernel[...] = tau * source.kernel[...] + (1 - tau) * destination.kernel[...]
        destination.bias[...] = tau * source.bias[...] + (1 - tau) * destination.bias[...]
