"""Step 4 (outcome): Shapley models, phi(s) explaining a frozen outcome characteristic.

The source's PerformanceShapley: one contribution per feature, of the outcome
characteristic v(e, C). Buffers are fresh rollouts of the behaviour characteristic's
explained policy; exact values use the behaviour characteristic's exact buffer, as the
outcome characteristic does. `with_exact_char` explains the exact outcome values (themselves
conditioned on the exact or learnt behaviour characteristic, as the outcome stage chose),
otherwise the learnt outcome characteristic. The behaviour stage, loss, initialisation and
exact Shapley values are imported (`shapley_models.behaviour`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array

from jax_shapley.algorithms.dqn import TRACKING_FIELDS, Agent, EnvParams, EnvState
from jax_shapley.characteristic_models import outcome
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeExplained, OutcomeFrozen
from jax_shapley.common.coalitions import coalition_table
from jax_shapley.common.networks import LayerConfig
from jax_shapley.common.rollouts import collect_buffers
from jax_shapley.common.supervised_learning import EpochMetrics, StateBuffers, TrainState, train
from jax_shapley.common.tabular import TabularModel, transition_model
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper
from jax_shapley.shapley_models.behaviour import (
    COALITION_BATCH,
    ExactShapley,
    ShapleyConfig,
    ShapleyContext,
    ShapleyModel,
    exact_shapley,
    initialize,
    loss,
    stage,
)


def main(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    characteristic: OutcomeConfig,
    agents: Agent,
    outcomes: OutcomeExplained,
    behaviour_buffers: StateBuffers,
) -> tuple[TrainState[ShapleyModel], EpochMetrics | None]:
    """Outcome Shapley models explaining frozen outcome characteristics (seed-stacked, one per Shapley seed)."""
    if config.buffers != "fresh":
        raise ValueError("outcome Shapley runs collect fresh buffers")
    tabular = transition_model(config.env) if config.train.exact_loss else None
    settings = {
        "characteristic": {name: value for name, value in asdict(explained).items() if name not in TRACKING_FIELDS},
        "outcome": {name: value for name, value in asdict(characteristic).items() if name not in TRACKING_FIELDS},
    }
    return stage(
        config,
        explained,
        agents,
        outcomes,
        behaviour_buffers,
        partial(fit, explained, characteristic, tabular),
        head,
        settings,
        "PerformanceShapley",
        "outcome",
    )


def fit(
    explained: CharacteristicConfig,
    characteristic: OutcomeConfig,
    tabular: TabularModel | None,
    config: ShapleyConfig,
    max_unique: int,
    log: Callable[[EpochMetrics], None] | None,
    keys: Array,
    agents: Agent,
    outcomes: OutcomeExplained,
    inherited: StateBuffers,
) -> tuple[TrainState[ShapleyModel], EpochMetrics]:
    """Train every seed on fresh rollouts; exact values come from the tabular MDP."""
    del max_unique

    def rollouts(
        config: ShapleyConfig,
        explained: CharacteristicConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        key: Array,
        agent: Agent,
        inherited: StateBuffers,
    ) -> tuple[Array, StateBuffers]:
        epsilon = agent.epsilon if explained.policy == "explore" else jnp.asarray(0.0, jnp.float32)
        training = config.train
        return collect_buffers(
            env,
            agent.model,
            epsilon,
            key,
            train_size=training.train_size,
            validation_size=training.validation_size,
            exact_size=training.exact_buffer_size,
            num_envs=training.num_envs,
            exact=inherited.exact,
        )

    collect = nnx.vmap(
        partial(rollouts, config, explained, EpisodeStatisticsWrapper(config.env)), in_axes=(0, 0, 0), out_axes=(0, 0)
    )
    keys, buffers = collect(keys, agents, inherited)
    state = initialize(config, explained, head, keys, buffers)
    features = state.buffers.train.shape[-1]

    def context(
        config: ShapleyConfig,
        explained: CharacteristicConfig,
        characteristic: OutcomeConfig,
        tabular: TabularModel | None,
        buffers: StateBuffers,
        outcome_explained: OutcomeExplained,
    ) -> ShapleyContext:
        if tabular is None:
            return ShapleyContext(buffers, None)
        frozen = OutcomeFrozen(outcome_explained.agent, outcome_explained.behaviour)
        exact = outcome.exact_outcome(
            tabular, buffers.exact, frozen, explained, characteristic.with_exact_char, characteristic.gamma
        )
        coalitions = jnp.asarray(coalition_table(exact.states.shape[-1]))
        values = (
            exact.values[..., None]
            if config.with_exact_char
            else jax.lax.map(
                partial(outcome.values, characteristic, explained, outcome_explained, exact.states),
                coalitions,
                batch_size=COALITION_BATCH,
            )
        )
        return ShapleyContext(buffers, ExactShapley(exact.states, exact.state_valid, exact_shapley(values)))

    contexts = nnx.vmap(partial(context, config, explained, characteristic, tabular), in_axes=(0, 0), out_axes=0)(
        state.buffers, outcomes
    )
    explained_loss = partial(
        loss, partial(outcome.values, characteristic, explained), partial(outcome.null, characteristic)
    )
    return train(state, contexts, outcomes, explained_loss, config.train, features, log)


def head(actions: int) -> LayerConfig:
    """One contribution per feature: the outcome characteristic is a single value."""
    del actions
    return LayerConfig(1)
