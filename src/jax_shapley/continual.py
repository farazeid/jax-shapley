"""Continual learning: explanation models trained alongside the agent (paper Figures 3b and 10).

`python -m jax_shapley.continual NAME [--track wandb|trackio] [overrides]` runs a config in
`jax_shapley/configs/NAME.py` whose CONFIG is a `ContinualConfig`. It ports upstream
`training.train_dqn` with explanation models (`experiments/parallel`). Every vector step,
in the source's order:

1. on exact steps, the exact buffer is refilled from fresh rollouts of the pre-update
   agent's explore policy (collected before the DQN step, which is equivalent: the
   rollouts start from a reset and the agent's epsilon is that of the current step);
2. the agent acts, stores the transitions and learns: `dqn.step`, unchanged;
3. `update_ratio` characteristic updates on fresh replay batches, importance-weighted
   by the current explore policy against the stored behaviour log-probabilities; for
   outcome runs, then `update_ratio` off-policy outcome updates;
4. `update_ratio` Shapley updates, one Shapley-kernel coalition per batch;
5. on exact steps, every model's exact loss against the current agent.

Explanation models learn whenever the agent does (`learning_starts`, `train_frequency`).
The agent's random stream is that of a plain DQN run with the same seed, so it trains
identically; explanation models use a separate stream. Seeds train together in one
compiled program. Each seed logs into its own run `NAME--continual--KIND--seed-N`, and the
seed-stacked final state (agents and explanation models) is pushed as a checkpoint artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from functools import partial
from math import isnan
from pathlib import Path
from time import time_ns
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import trackio
import tyro
import wandb
from flax import nnx, struct
from jax import Array

from jax_shapley import backends
from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour as characteristic
from jax_shapley.characteristic_models import outcome, prediction
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicModel, Values
from jax_shapley.characteristic_models.outcome import (
    OutcomeConfig,
    OutcomeExplained,
    OutcomeFrozen,
    OutcomeNetworks,
)
from jax_shapley.common import checkpoints
from jax_shapley.common.coalitions import (
    Coalition,
    Coalitions,
    Observations,
    coalition_table,
    kernel_coalitions,
    mask,
)
from jax_shapley.common.networks import LayerConfig, MLPConfig, soft_update
from jax_shapley.common.rollouts import collect_observations
from jax_shapley.common.tabular import TabularModel, transition_model
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper
from jax_shapley.experiment import ContinualConfig, load_configs
from jax_shapley.shapley_models.behaviour import (
    COALITION_BATCH,
    CharacteristicNull,
    CharacteristicValues,
    ShapleyModel,
    exact_shapley,
    masked_null,
    masked_values,
)
from jax_shapley.shapley_models.behaviour import explain as explain_values

type Explained = CharacteristicModel | OutcomeExplained
"""What a continual Shapley model explains: the learnt characteristic, or the learnt outcome characteristic."""

# Explanation streams fold this into each seed's key; the agent keeps the plain DQN stream.
EXPLANATION_STREAM = 1


@struct.dataclass
class Explainers:
    """One seed's explanation models; the outcome networks exist only for outcome runs."""

    characteristic: CharacteristicModel
    characteristic_optimizer: nnx.Optimizer[CharacteristicModel]
    shapley: ShapleyModel
    shapley_optimizer: nnx.Optimizer[ShapleyModel]
    outcome: OutcomeNetworks | None
    outcome_optimizer: nnx.Optimizer[OutcomeNetworks] | None


@struct.dataclass
class ContinualState:
    run: dqn.RunState
    explainers: Explainers
    exact: Array
    key: Array


@struct.dataclass
class ContinualMetrics:
    """One vector step of every seed: the agent's metrics, each explanation model's loss on its last
    update of the step, and exact losses; NaN where not computed."""

    agent: dqn.StepMetrics
    characteristic_loss: Array
    outcome_loss: Array
    shapley_loss: Array
    characteristic_exact: Array
    outcome_exact: Array
    shapley_exact: Array
    logged: Array
    exact_step: Array


def main(args: list[str] | None = None) -> str:
    """Run the named continual experiment, tracked; returns its run group (empty when only compiling)."""
    config = tyro.extras.overridable_config_cli(load_configs(ContinualConfig), args=args)
    tabular = transition_model(config.algorithm.env)
    if config.compile_only:
        seeds = jax.ShapeDtypeStruct((len(config.algorithm.seeds),), jax.random.key(0).dtype)
        nnx.jit(partial(fit, config, tabular, lambda metrics: None)).trace(seeds).lower().compile()
        print(f"Compiled {config.name}.")
        return ""
    group = f"{config.name}-{time_ns()}"
    kind = config.explain.kind
    artifact = f"{group}--continual--{kind}"
    settings = {
        "stage": "continual",
        "env": config.algorithm.env.name,
        "config": {
            name: value
            for name, value in asdict(config).items()
            if name not in ("checkpoint", "compile_only", "track", "project", "entity")
        },
    }
    keys = jnp.stack([jax.random.key(seed) for seed in config.algorithm.seeds])
    with backends.session(config.track, config.entity) as root:
        tracking = wandb if config.track == "wandb" else trackio
        runs = tuple(
            tracking.init(
                project=config.project,
                name=f"{config.name}--continual--{kind}--seed-{seed}",
                group=group,
                config={**asdict(config), "seed": seed},
            )
            for seed in config.algorithm.seeds
        )
        try:
            if config.checkpoint is not None:
                directory = runs[0].use_artifact(config.checkpoint).download(root=str(root / "artifacts" / artifact))
                checkpoints.load(Path(directory), initialize(config, keys), settings)
                return group
            state, _ = nnx.jit(partial(fit, config, tabular, partial(log, runs, config.algorithm.num_envs)))(keys)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            backends.upload(runs[0], directory, artifact, "continual")
            print(f"Continual checkpoint: {artifact}:latest")
        finally:
            for run in runs:
                run.finish()
    print(f"Finished {config.name}: {config.track} group {group} in project {config.project}.")
    return group


def fit(
    config: ContinualConfig,
    tabular: TabularModel,
    log: Callable[[ContinualMetrics], None] | None,
    keys: Array,
) -> tuple[ContinualState, ContinualMetrics]:
    """Every seed's agent and explanations over the whole run; metrics carry (step, seed) axes.

    Shared step predicates stay outside the seed vmap, so their conditions do not become
    selects; NNX graph nodes travel in the carries, as transforms accept them only at
    their own trace level.
    """
    state = initialize(config, keys)
    env = EpisodeStatisticsWrapper(config.algorithm.env)
    training, explain = config.algorithm, config.explain

    def step(
        config: ContinualConfig,
        tabular: TabularModel,
        env: EpisodeStatisticsWrapper[dqn.EnvState, dqn.EnvParams],
        carry: tuple[ContinualState, Array],
        index: Array,
    ) -> tuple[tuple[ContinualState, Array], ContinualMetrics]:
        del index
        state, vector_step = carry
        training, explain = config.algorithm, config.explain
        global_step = vector_step * training.num_envs
        learn = (global_step > training.learning_starts) & (global_step % training.train_frequency < training.num_envs)
        exact_step = learn & (global_step % explain.exact_frequency < training.num_envs)
        logged = learn & (global_step % dqn.LOG_STEPS < training.num_envs)
        fraction = global_step.astype(jnp.float32) / (training.exploration_fraction * training.total_timesteps)
        epsilon = jnp.maximum(training.start_e + (training.end_e - training.start_e) * fraction, training.end_e)
        epsilon = epsilon.astype(jnp.float32)
        state = nnx.cond(
            exact_step,
            nnx.vmap(partial(refresh_exact, config, env), in_axes=(0, None), out_axes=0),
            lambda state, epsilon: state,
            state,
            epsilon,
        )
        state, agent = nnx.vmap(partial(agent_step, config, env), in_axes=(0, 0), out_axes=(0, 0))(
            state, jnp.arange(keys.shape[0], dtype=jnp.int32)
        )
        state, losses = nnx.cond(
            learn,
            nnx.vmap(partial(explanation_updates, config), in_axes=(0, None), out_axes=(0, 0)),
            skip_updates,
            state,
            epsilon,
        )
        exact = nnx.cond(
            exact_step,
            nnx.vmap(partial(exact_losses, config, tabular), in_axes=(0, None), out_axes=0),
            skip_exact,
            state,
            epsilon,
        )
        seeds = keys.shape[0]
        metrics = ContinualMetrics(
            agent,
            losses[0],
            losses[1],
            losses[2],
            exact[0],
            exact[1],
            exact[2],
            jnp.broadcast_to(logged, (seeds,)),
            jnp.broadcast_to(exact_step, (seeds,)),
        )
        return (state, vector_step + 1), metrics

    def block(
        config: ContinualConfig,
        tabular: TabularModel,
        log: Callable[[ContinualMetrics], None] | None,
        env: EpisodeStatisticsWrapper[dqn.EnvState, dqn.EnvParams],
        carry: tuple[ContinualState, Array],
        indices: Array,
    ) -> tuple[tuple[ContinualState, Array], ContinualMetrics]:
        """Consecutive vector steps, then one live log of them all."""
        carry, metrics = nnx.scan(partial(step, config, tabular, env), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            carry, indices
        )
        if log is not None:
            jax.debug.callback(log, metrics, ordered=True)
        return carry, metrics

    carry = (state, jnp.asarray(0, jnp.int32))
    full, rest = divmod(training.vector_steps, dqn.STEP_BLOCK)
    pieces: list[ContinualMetrics] = []
    if full:
        indices = jnp.arange(full * dqn.STEP_BLOCK, dtype=jnp.int32).reshape(full, dqn.STEP_BLOCK)
        carry, blocked = nnx.scan(
            partial(block, config, tabular, log, env), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )(carry, indices)
        pieces.append(jax.tree.map(lambda leaf: leaf.reshape(full * dqn.STEP_BLOCK, *leaf.shape[2:]), blocked))
    if rest:
        carry, tail = block(config, tabular, log, env, carry, jnp.arange(rest, dtype=jnp.int32))
        pieces.append(tail)
    state, _ = carry
    columns = zip(*[jax.tree.leaves(piece) for piece in pieces], strict=True)
    leaves = [jnp.concatenate(column) for column in columns]
    del explain
    # Unflattening with the pieces' own structure rebuilds the struct.
    return state, cast(ContinualMetrics, jax.tree.unflatten(jax.tree.structure(pieces[0]), leaves))


def initialize(config: ContinualConfig, keys: Array) -> ContinualState:
    """Every seed's fresh agent (the plain DQN initialisation) and explanation models with Adam
    optimisers, from a separate explanation stream; also a checkpoint's template."""
    env = EpisodeStatisticsWrapper(config.algorithm.env)
    shape = env.observation_space(env.default_params).shape
    features, actions = int(shape[0]), env.num_actions
    explain = config.explain
    hidden = tuple(LayerConfig(width, "relu") for width in explain.hidden_widths)
    head = LayerConfig(actions, "softmax") if explain.kind != "prediction" else LayerConfig(1)
    outputs = actions if explain.kind == "behaviour" else 1

    def seed(config: ContinualConfig, key: Array) -> ContinualState:
        explain = config.explain
        run = dqn.initialize(config.algorithm, key)
        model_keys: Array = jax.random.split(jax.random.fold_in(key, EXPLANATION_STREAM), 4)
        adam = optax.adam(
            explain.learning_rate, b1=explain.adam.beta1, b2=explain.adam.beta2, eps=explain.adam.epsilon, eps_root=0.0
        )
        model = CharacteristicModel(MLPConfig(features, (*hidden, head)), rngs=nnx.Rngs(model_keys[1]))
        shapley = ShapleyModel(
            MLPConfig(features, (*hidden, LayerConfig(features * outputs))), features, rngs=nnx.Rngs(model_keys[2])
        )
        networks, networks_optimizer = None, None
        if explain.kind == "outcome":
            network = MLPConfig(3 * features, (*hidden, LayerConfig(actions)))
            networks = OutcomeNetworks(network, rngs=nnx.Rngs(model_keys[3]))
            networks_optimizer = nnx.Optimizer(networks, adam, wrt=nnx.Param)
        explainers = Explainers(
            model,
            nnx.Optimizer(model, adam, wrt=nnx.Param),
            shapley,
            nnx.Optimizer(shapley, adam, wrt=nnx.Param),
            networks,
            networks_optimizer,
        )
        exact = jnp.zeros((explain.exact_buffer_size, features), jnp.int32)
        return ContinualState(run, explainers, exact, model_keys[0])

    return nnx.vmap(partial(seed, config), in_axes=0, out_axes=0)(keys)


def refresh_exact(
    config: ContinualConfig,
    env: EpisodeStatisticsWrapper[dqn.EnvState, dqn.EnvParams],
    state: ContinualState,
    epsilon: Array,
) -> ContinualState:
    """Source fill_buffer(dqn, copy(envs), exact_buffer, size, 'explore'): fresh explore rollouts."""
    keys: Array = jax.random.split(state.key)
    exact = collect_observations(
        env, state.run.model, epsilon, keys[1], config.explain.exact_buffer_size, config.algorithm.num_envs
    )
    return ContinualState(state.run, state.explainers, exact.astype(jnp.int32), keys[0])


def agent_step(
    config: ContinualConfig,
    env: EpisodeStatisticsWrapper[dqn.EnvState, dqn.EnvParams],
    state: ContinualState,
    seed: Array,
) -> tuple[ContinualState, dqn.StepMetrics]:
    """The unchanged DQN step."""
    del seed
    run, metrics = dqn.step(config.algorithm, env, state.run, jnp.asarray(0, jnp.int32))
    return ContinualState(run, state.explainers, state.exact, state.key), metrics


def explanation_updates(
    config: ContinualConfig, state: ContinualState, epsilon: Array
) -> tuple[ContinualState, tuple[Array, Array, Array]]:
    """`update_ratio` updates of each explanation model, in the source's model order; each model's
    loss on its last update."""
    explain, keys = config.explain, jax.random.split(state.key, 4)
    agent = dqn.Agent(state.run.model, epsilon)
    type Carry = tuple[Explainers, dqn.Agent, dqn.ReplayState]

    def repeat(
        update: Callable[[ContinualConfig, Carry, Array], tuple[Carry, Array]], carry: Carry, key: Array
    ) -> tuple[Carry, Array]:
        carry, losses = nnx.scan(partial(update, config), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            carry, jax.random.split(key, explain.update_ratio)
        )
        return carry, losses[-1]

    carry: Carry = (state.explainers, agent, state.run.replay)
    carry, characteristic_loss = repeat(characteristic_update, carry, keys[1])
    outcome_loss = jnp.asarray(jnp.nan, jnp.float32)
    if explain.kind == "outcome":
        carry, outcome_loss = repeat(outcome_update, carry, keys[2])
    carry, shapley_loss = repeat(shapley_update, carry, keys[3])
    return ContinualState(state.run, carry[0], state.exact, keys[0]), (characteristic_loss, outcome_loss, shapley_loss)


def characteristic_update(
    config: ContinualConfig, carry: tuple[Explainers, dqn.Agent, dqn.ReplayState], key: Array
) -> tuple[tuple[Explainers, dqn.Agent, dqn.ReplayState], Array]:
    """One importance-weighted Adam step of the characteristic on a fresh replay batch: rows weigh
    by the current explore policy against the stored behaviour log-probabilities."""
    explainers, agent, replay = carry
    explain, explained = config.explain, explained_config(config)
    batch_key, coalition_key = jax.random.split(key)
    batch = sample(replay, config.algorithm.batch_size, batch_key)
    targets = (prediction.targets if explain.kind == "prediction" else characteristic.targets)(
        explained, agent, batch.observations
    )
    # The explore policy's clamped log-probability of each stored action (Torch Categorical).
    q_values = agent.model(batch.observations).q_values
    ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
    greedy = ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)
    probabilities = agent.epsilon / q_values.shape[-1] + (1 - agent.epsilon) * greedy
    explore = dqn.log_probabilities(probabilities, batch.actions)
    weights = jnp.exp(explore - batch.log_probabilities)
    if explain.weighting:
        weights = weights / jnp.sum(weights)
    weights = jnp.clip(weights, 1 - explain.clip_coef, 1 + explain.clip_coef)
    coalitions = jax.random.bernoulli(coalition_key, 0.5, (batch.observations.shape[0], batch.observations.shape[-1]))

    def loss(
        model: CharacteristicModel, observations: Array, targets: Array, coalitions: Array, weights: Array
    ) -> Array:
        predictions = model(mask(observations, coalitions)).values
        errors = jnp.square(predictions - jax.lax.stop_gradient(targets))
        return jnp.mean(jax.lax.stop_gradient(weights)[:, None] * errors)

    # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
    value, gradients = cast(
        tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(loss)(explainers.characteristic, batch.observations, targets, coalitions, weights),
    )
    explainers.characteristic_optimizer.update(explainers.characteristic, gradients)
    return carry, value


def outcome_update(
    config: ContinualConfig, carry: tuple[Explainers, dqn.Agent, dqn.ReplayState], key: Array
) -> tuple[tuple[Explainers, dqn.Agent, dqn.ReplayState], Array]:
    """One off-policy outcome Q-learning step on replay transitions and independently sampled
    explained rows, then the soft target update."""
    explainers, agent, replay = carry
    networks, optimizer = explainers.outcome, explainers.outcome_optimizer
    assert networks is not None and optimizer is not None
    explain, explained = config.explain, explained_config(config)
    batch_key, explained_key, coalition_key, action_key = jax.random.split(key, 4)
    size = config.algorithm.batch_size
    batch = sample(replay, size, batch_key)
    explained_rows = sample(replay, size, explained_key).observations
    coalitions = jax.random.bernoulli(coalition_key, 0.5, (size, explained_rows.shape[-1]))
    behaviour = explainers.characteristic(mask(explained_rows, coalitions)).values
    policy = outcome.conditioned_policy(explained, agent, behaviour, batch.next_observations, explained_rows)
    next_actions = jax.random.categorical(action_key, jnp.log(policy), axis=-1).astype(jnp.int32)

    def loss(
        networks: OutcomeNetworks,
        batch: dqn.TransitionBatch,
        explained_rows: Array,
        coalitions: Array,
        next_actions: Array,
        gamma: float,
    ) -> Array:
        target_values = networks.target(batch.next_observations, explained_rows, coalitions).values
        following = jnp.take_along_axis(target_values, next_actions[:, None], axis=-1)[:, 0]
        alive = (~batch.terminated).astype(jnp.float32)
        targets = jax.lax.stop_gradient(batch.rewards + gamma * following * alive)
        values = networks.model(batch.observations, explained_rows, coalitions).values
        chosen = jnp.take_along_axis(values, batch.actions[:, None], axis=-1)[:, 0]
        return jnp.mean(jnp.square(chosen - targets))

    # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
    value, gradients = cast(
        tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(loss)(networks, batch, explained_rows, coalitions, next_actions, explain.gamma),
    )
    optimizer.update(networks, gradients)
    soft_update(networks.model.mlp, networks.target.mlp, explain.tau)
    return carry, value


def shapley_update(
    config: ContinualConfig, carry: tuple[Explainers, dqn.Agent, dqn.ReplayState], key: Array
) -> tuple[tuple[Explainers, dqn.Agent, dqn.ReplayState], Array]:
    """One Adam step of the Shapley model on a replay batch, one Shapley-kernel coalition for the batch,
    explaining the learnt characteristic (or the learnt outcome characteristic)."""
    explainers, agent, replay = carry
    batch_key, coalition_key = jax.random.split(key)
    observations = sample(replay, config.algorithm.batch_size, batch_key).observations
    coalition = kernel_coalitions(coalition_key, 1, observations.shape[-1])[0]
    values, null, frozen = explained_function(config, explainers, agent)

    def loss(
        values: CharacteristicValues[Explained],
        null: CharacteristicNull[Explained],
        model: ShapleyModel,
        frozen: Explained,
        observations: Observations,
        coalition: Coalition,
    ) -> Array:
        targets = jax.lax.stop_gradient(values(frozen, observations, coalition) - null(frozen, observations))
        predictions = jnp.sum(coalition[:, None].astype(jnp.float32) * model(observations).contributions, axis=-2)
        return jnp.mean(jnp.square(predictions - targets))

    # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
    value, gradients = cast(
        tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(partial(loss, values, null))(explainers.shapley, frozen, observations, coalition),
    )
    explainers.shapley_optimizer.update(explainers.shapley, gradients)
    return carry, value


def exact_losses(
    config: ContinualConfig, tabular: TabularModel, state: ContinualState, epsilon: Array
) -> tuple[Array, Array, Array]:
    """Every model's exact loss against the current agent on the refreshed exact buffer."""
    explain, explained = config.explain, explained_config(config)
    explainers, agent = state.explainers, dqn.Agent(state.run.model, epsilon)
    observations = state.exact
    coalitions = jnp.asarray(coalition_table(observations.shape[-1]))
    targets = (prediction.targets if explain.kind == "prediction" else characteristic.targets)(
        explained, agent, observations
    )
    exact = characteristic.exact_characteristic(
        observations, targets, coalitions, max_unique=int(tabular.states.shape[0])
    )
    characteristic_exact = characteristic.exact_loss(explainers.characteristic, exact)
    values, null, frozen = explained_function(config, explainers, agent)
    if explain.kind != "outcome":
        table = (
            exact.at_states
            if explain.with_exact_char
            else jax.lax.map(
                partial(masked_values, explainers.characteristic, exact.states), coalitions, batch_size=COALITION_BATCH
            )
        )
        states, valid, outcome_exact = exact.states, exact.state_valid, jnp.asarray(jnp.nan, jnp.float32)
    else:
        assert isinstance(frozen, OutcomeExplained)
        outcome_config = outcome_settings(config)
        exact_outcome = outcome.exact_outcome(
            tabular,
            observations,
            OutcomeFrozen(agent, explainers.characteristic),
            explained,
            explain.outcome_with_exact_char,
            explain.gamma,
        )
        outcome_exact = outcome.exact_values_loss(outcome_config, explained, frozen, exact_outcome)
        table = (
            exact_outcome.values[..., None]
            if explain.with_exact_char
            else jax.lax.map(
                partial(outcome.values, outcome_config, explained, frozen, exact_outcome.states),
                coalitions,
                batch_size=COALITION_BATCH,
            )
        )
        states, valid = exact_outcome.states, exact_outcome.state_valid
    errors = jnp.square(explain_values(values, null, explainers.shapley, frozen, states) - exact_shapley(table))
    weights = valid.astype(jnp.float32)
    shapley_exact = jnp.sum(errors * weights[:, None, None]) / (jnp.sum(weights) * errors.shape[-2] * errors.shape[-1])
    return characteristic_exact, outcome_exact, shapley_exact


def skip_updates(state: ContinualState, epsilon: Array) -> tuple[ContinualState, tuple[Array, Array, Array]]:
    """Steps without learning: no losses."""
    del epsilon
    missing = jnp.full(state.key.shape, jnp.nan, jnp.float32)
    return state, (missing, missing, missing)


def skip_exact(state: ContinualState, epsilon: Array) -> tuple[Array, Array, Array]:
    """Steps without exact losses."""
    del epsilon
    missing = jnp.full(state.key.shape, jnp.nan, jnp.float32)
    return missing, missing, missing


def explained_config(config: ContinualConfig) -> CharacteristicConfig:
    """The explained characteristic: the agent's explore policy (behaviour, also for outcome) or value."""
    kind = "prediction" if config.explain.kind == "prediction" else "behaviour"
    return CharacteristicConfig(
        config.algorithm.env, kind=kind, policy="explore", hidden_widths=config.explain.hidden_widths
    )


def outcome_settings(config: ContinualConfig) -> OutcomeConfig:
    """The off-policy outcome characteristic's settings."""
    explain = config.explain
    return OutcomeConfig(
        config.algorithm.env,
        variant="off_policy",
        with_exact_char=explain.outcome_with_exact_char,
        tau=explain.tau,
        gamma=explain.gamma,
        hidden_widths=explain.hidden_widths,
    )


def explained_function(
    config: ContinualConfig, explainers: Explainers, agent: dqn.Agent
) -> tuple[CharacteristicValues[Explained], CharacteristicNull[Explained], Explained]:
    """What the Shapley model explains: the learnt characteristic, or the learnt outcome characteristic."""
    if explainers.outcome is None:
        frozen: Explained = explainers.characteristic
    else:
        frozen = OutcomeExplained(agent, explainers.characteristic, explainers.outcome)
    return partial(characteristic_values, config), partial(characteristic_null, config), frozen


def characteristic_values(
    config: ContinualConfig, frozen: Explained, observations: Observations, coalitions: Coalitions
) -> Values:
    """v(s, C) of the learnt characteristic, or of the learnt (off-policy) outcome characteristic."""
    if isinstance(frozen, OutcomeExplained):
        return outcome.values(outcome_settings(config), explained_config(config), frozen, observations, coalitions)
    return masked_values(frozen, observations, coalitions)


def characteristic_null(config: ContinualConfig, frozen: Explained, observations: Observations) -> Values:
    """v(s, empty) of the learnt characteristic, or of the learnt (off-policy) outcome characteristic."""
    if isinstance(frozen, OutcomeExplained):
        return outcome.null(outcome_settings(config), frozen, observations)
    return masked_null(frozen, observations)


def sample(replay: dqn.ReplayState, size: int, key: Array) -> dqn.TransitionBatch:
    """Uniform replay rows with replacement from the filled prefix (the source Buffer.sample)."""
    keys: Array = jax.random.split(key)
    rows = jax.random.randint(keys[1], (size,), 0, replay.size, dtype=jnp.int32)
    return dqn.TransitionBatch(*[leaf[rows] for leaf in jax.tree.leaves(replay.data)])


def log(runs: Sequence[backends.Logger], num_envs: int, metrics: ContinualMetrics) -> None:
    """A block of vector steps, (steps, seeds), into each seed's run (a host callback): the agent's
    source scalars, and on logged and exact steps the explanation models' losses."""
    agent = metrics.agent
    steps, completed = np.asarray(agent.global_step), np.asarray(agent.completed)
    returns, lengths, epsilons = (
        np.asarray(agent.episode_return),
        np.asarray(agent.episode_length),
        np.asarray(agent.epsilon),
    )
    logged, exact_steps, losses = np.asarray(metrics.logged), np.asarray(metrics.exact_step), np.asarray(agent.loss)
    models = {
        name: np.asarray(getattr(metrics, name)) for name in ("characteristic_loss", "outcome_loss", "shapley_loss")
    }
    exact = {
        name: np.asarray(getattr(metrics, name)) for name in ("characteristic_exact", "outcome_exact", "shapley_exact")
    }
    for index in np.ndindex(steps.shape):
        row: dict[str, float] = {}
        if completed[index].any():
            row["agent/episodic_return"] = float(returns[index][completed[index]].mean())
            row["agent/episodic_length"] = float(lengths[index][completed[index]].mean())
            row["agent/epsilon"] = float(epsilons[index])
        if bool(logged[index]):
            row["agent/DQN_loss"] = float(losses[index])
            row |= {
                f"model/{name}": float(values[index]) for name, values in models.items() if not isnan(values[index])
            }
        if bool(exact_steps[index]):
            row |= {
                f"model/{name}_loss": float(values[index]) for name, values in exact.items() if not isnan(values[index])
            }
        if row:
            runs[index[1]].log(row, step=int(steps[index]))
    del num_envs


if __name__ == "__main__":
    main()
