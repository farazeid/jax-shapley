"""Continual learning: explanation models trained alongside the agent (paper Figures 3b and 10).

Port of upstream `training.train_dqn` with explanation models (`experiments/parallel`).
Every vector step, in the source's order:

1. the agent acts and stores the transitions (the unchanged DQN `training_step`, which
   also takes its one learning step and soft target update);
2. on exact steps, the exact buffer is refilled from fresh rollouts of the pre-update
   agent's explore policy (collected before the DQN step, which is equivalent: the
   rollouts start from a reset and the agent's epsilon is that of the current step);
3. `update_ratio` characteristic updates on fresh replay batches, importance-weighted
   by the current explore policy against the stored behaviour log-probabilities;
   for outcome runs, then `update_ratio` off-policy outcome updates;
4. `update_ratio` Shapley updates, one Shapley-kernel coalition per batch;
5. on exact steps, every model's exact loss against the current agent.

Explanation models learn whenever the agent does (`learning_starts`, `train_frequency`).
The agent's random stream is that of a plain DQN run with the same seed, so the agent
trains identically; explanation models use a separate stream. Seeds train together in
one compiled program, and the run returns every step's metrics.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from math import inf, isfinite, nan
from pathlib import Path
from time import perf_counter, time_ns
from typing import Annotated, Literal, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import tyro
from flax import nnx, struct
from jax import Array

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    OffPolicyConfig,
    characteristic_network,
    exact_characteristic,
    exact_characteristic_loss,
    explained_values,
    off_policy_weights,
    policy_probabilities,
    update_characteristic,
)
from jax_shapley.coalitions import all_coalitions, kernel_coalitions, uniform_coalitions
from jax_shapley.networks import QNetwork
from jax_shapley.outcome import (
    AgentPolicy,
    ConditionedPolicy,
    OutcomeCharacteristic,
    OutcomeConfig,
    OutcomeNetworks,
    Transitions,
    exact_outcome_loss,
    outcome_network,
    sample_actions,
    update_off_policy,
)
from jax_shapley.replay import TransitionBatch, sample
from jax_shapley.shapley import (
    CharacteristicFunction,
    MaskedCharacteristic,
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    shapley_network,
    update_shapley,
)
from jax_shapley.tabular import TabularModel, transition_model
from jax_shapley.train_dqn import (
    AdamConfig,
    DQNTrainConfig,
    GridworldConfig,
    Mastermind222Config,
    RunState,
    StepMetrics,
    action_log_probabilities,
    epsilon_at,
    initialize_run,
    make_adam,
    network_config,
    training_step,
)
from jax_shapley.train_explanations import FrozenAgent, collect_observations, outcome_exact
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsWrapper

type ContinualKind = Literal["behaviour", "prediction", "outcome"]
type ContinualEnvironment = GridworldConfig | Mastermind222Config

# Source logging cadence for training losses, in environment steps.
LOG_STEPS = 100
# Explanation streams fold this into each seed's key; the agent keeps the plain DQN stream.
EXPLANATION_STREAM = 1


@dataclass(frozen=True, slots=True)
class ContinualConfig:
    """Explanation-side settings of the source's parallel drivers.

    `with_exact_char` selects the Shapley ground truth (exact characteristic: the w_exact
    and w_policy groups); `outcome_with_exact_char` conditions exact outcome values on the
    exact behaviour characteristic (w_policy).
    """

    kind: ContinualKind = "behaviour"
    update_ratio: int = 1
    with_exact_char: bool = True
    outcome_with_exact_char: bool = True
    exact_frequency: int = 1_000
    exact_buffer_size: int = 10_000
    hidden_widths: tuple[int, ...] = (64, 64)
    learning_rate: float = 2.5e-4
    tau: float = 0.01
    gamma: float = 1.0
    weighting: bool = True
    clip_coef: float = inf
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        if self.kind not in ("behaviour", "prediction", "outcome"):
            raise ValueError("kind must be behaviour, prediction or outcome")
        for name, value in (
            ("update_ratio", self.update_ratio),
            ("exact_frequency", self.exact_frequency),
            ("exact_buffer_size", self.exact_buffer_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        CharacteristicConfig("behaviour", "explore", self.hidden_widths)
        OffPolicyConfig(self.weighting, self.clip_coef)
        OutcomeConfig("off_policy", self.tau, self.gamma, self.hidden_widths)

    @property
    def characteristic(self) -> CharacteristicConfig:
        """The explained characteristic: the agent's explore policy (behaviour, also for outcome) or value."""
        kind = "prediction" if self.kind == "prediction" else "behaviour"
        return CharacteristicConfig(kind, "explore", self.hidden_widths)

    @property
    def importance(self) -> OffPolicyConfig:
        return OffPolicyConfig(self.weighting, self.clip_coef)

    @property
    def outcome(self) -> OutcomeConfig:
        return OutcomeConfig("off_policy", self.tau, self.gamma, self.hidden_widths)


@dataclass(frozen=True, slots=True)
class ContinualRunConfig[Environment: ContinualEnvironment]:
    """A continual sweep: one agent and its explanation models per seed."""

    env: Environment
    train: DQNTrainConfig = field(default_factory=DQNTrainConfig)
    explain: ContinualConfig = field(default_factory=ContinualConfig)
    hidden_widths: tuple[int, ...] = (64, 64)
    seeds: tuple[int, ...] = (1,)
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct and nonempty")
        if any(not 0 <= seed <= 2**32 - 1 for seed in self.seeds):
            raise ValueError("seeds must be in [0, 2**32 - 1]")
        if self.explain.exact_buffer_size % self.train.num_envs:
            raise ValueError("exact_buffer_size must be a multiple of num_envs")
        if self.validate_only and self.compile_only:
            raise ValueError("validate_only and compile_only are mutually exclusive")


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
class ContinualState[EnvState]:
    run: RunState[EnvState]
    explainers: Explainers
    exact: Array
    key: Array


@struct.dataclass
class ExplanationLosses:
    """Each model's loss on its last update of the step (the source logs the last)."""

    characteristic: Array
    outcome: Array
    shapley: Array


@struct.dataclass
class ExactLosses:
    characteristic: Array
    outcome: Array
    shapley: Array


@struct.dataclass
class ContinualMetrics:
    """One vector step of one seed; losses are NaN when not computed."""

    agent: StepMetrics
    losses: ExplanationLosses
    exact: ExactLosses
    logged: Array
    exact_step: Array


@dataclass(frozen=True, slots=True)
class ContinualSummary:
    folder: str
    seed: int
    steps: int
    seconds: float


class ContinualRunner(Protocol):
    def __call__(self, keys: Array) -> tuple[ContinualState[object], ContinualMetrics]: ...


type CLIConfig = (
    Annotated[
        ContinualRunConfig[GridworldConfig],
        tyro.conf.subcommand(name="gridworld", default=ContinualRunConfig(GridworldConfig())),
    ]
    | Annotated[
        ContinualRunConfig[Mastermind222Config],
        tyro.conf.subcommand(name="mastermind222", default=ContinualRunConfig(Mastermind222Config())),
    ]
)


def main(args: list[str] | None = None) -> list[ContinualSummary] | None:
    """Validate, compile, run every seed once and write each seed's metrics."""
    config = parse_args(args)
    if config.validate_only:
        print(json.dumps(_settings(config), indent=2, default=str))
        return None
    tabular = transition_model(config.env)
    runner = nnx.jit(partial(run_continual, config=config, tabular=tabular))
    keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    compiled = cast(
        ContinualRunner,
        runner.trace(jax.ShapeDtypeStruct(keys.shape, keys.dtype)).lower().compile(),
    )
    if config.compile_only:
        print(f"Compiled {config.env.make().name} continual {config.explain.kind} on {jax.default_backend()}.")
        return None
    start = perf_counter()
    _, metrics = compiled(keys)
    jax.block_until_ready(metrics)
    seconds = perf_counter() - start
    root = config.output / f"{config.env.make().name}-continual-{config.explain.kind}-{time_ns()}"
    summaries = []
    for index, seed in enumerate(config.seeds):
        folder = root / f"seed-{seed}"
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(
            json.dumps(_settings(replace(config, seeds=(seed,))), indent=2, default=str)
        )
        write_metrics(folder, seed_slice(metrics, index))
        summaries.append(ContinualSummary(str(folder), seed, config.train.vector_steps, seconds))
        print(f"Trained {config.env.make().name} continual {config.explain.kind} seed {seed}. Artifacts: {folder}")
    return summaries


def run_continual(
    keys: Array, *, config: ContinualRunConfig[ContinualEnvironment], tabular: TabularModel
) -> tuple[ContinualState[object], ContinualMetrics]:
    """Every seed's agent and explanations over the whole run; metrics carry (step, seed) axes."""
    env = EpisodeStatisticsWrapper(cast(EnvironmentProtocol[object, object], config.env.make()))
    params = env.default_params
    training, explain = config.train, config.explain
    network = network_config(config.env, config.hidden_widths)
    features = network.input_features
    actions = network.output_features
    max_unique = int(tabular.states.shape[0])

    def initialize(key: Array) -> ContinualState[object]:
        run = initialize_run(env, params, network, training, key)
        explanation_key = jax.random.fold_in(key, EXPLANATION_STREAM)
        model_keys: Array = jax.random.split(explanation_key, 4)
        explainers = initialize_explainers(model_keys[1:], features, actions, explain)
        exact = jnp.zeros((explain.exact_buffer_size, features), jnp.int32)
        return ContinualState(run, explainers, exact, model_keys[0])

    state = nnx.vmap(initialize, in_axes=0, out_axes=0)(keys)

    def step(
        carry: tuple[ContinualState[object], Array], index: Array
    ) -> tuple[tuple[ContinualState[object], Array], ContinualMetrics]:
        del index
        state, vector_step = carry
        global_step = vector_step * training.num_envs
        learn = (global_step > training.learning_starts) & (global_step % training.train_frequency < training.num_envs)
        # Shared step predicates stay outside the seed vmap, so their conditions do not become selects.
        exact_step = learn & (global_step % explain.exact_frequency < training.num_envs)
        logged = learn & (global_step % LOG_STEPS < training.num_envs)
        epsilon = epsilon_at(training, global_step)
        # NNX cond keeps the carry topology; the casts restate it for the checker.
        state = cast(
            ContinualState[object],
            nnx.cond(
                exact_step,
                lambda value: nnx.vmap(partial(refresh_exact, env=env, epsilon=epsilon, config=config))(value),
                lambda value: value,
                state,
            ),
        )
        state, agent = nnx.vmap(partial(agent_step, env=env, params=params, training=training))(state)
        state, losses = cast(
            tuple[ContinualState[object], ExplanationLosses],
            nnx.cond(
                learn,
                lambda value: nnx.vmap(partial(explanation_updates, epsilon=epsilon, config=config, features=features))(
                    value
                ),
                lambda value: (value, _no_losses(value.key.shape)),
                state,
            ),
        )
        exact = nnx.cond(
            exact_step,
            lambda value: nnx.vmap(
                partial(exact_losses, epsilon=epsilon, explain=explain, tabular=tabular, max_unique=max_unique)
            )(value),
            lambda value: _no_exact(value.key.shape),
            state,
        )
        seeds = keys.shape[0]
        metrics = ContinualMetrics(
            agent,
            losses,
            exact,
            jnp.broadcast_to(logged, (seeds,)),
            jnp.broadcast_to(exact_step, (seeds,)),
        )
        return (state, vector_step + 1), metrics

    (state, _), metrics = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
        (state, jnp.asarray(0, jnp.int32)), jnp.arange(training.vector_steps, dtype=jnp.int32)
    )
    return state, metrics


def parse_args(args: list[str] | None = None) -> ContinualRunConfig[ContinualEnvironment]:
    return cast(ContinualRunConfig[ContinualEnvironment], tyro.cli(CLIConfig, args=args))


def initialize_explainers(keys: Array, features: int, actions: int, explain: ContinualConfig) -> Explainers:
    """Fresh characteristic, Shapley and (outcome runs) outcome models with Adam optimisers."""
    explained = explain.characteristic
    characteristic = CharacteristicModel(characteristic_network(features, actions, explained), rngs=nnx.Rngs(keys[0]))
    outputs = actions if explain.kind == "behaviour" else 1
    layout = ShapleyLayout(features, outputs)
    shapley = ShapleyModel(shapley_network(layout, explain.hidden_widths), layout, rngs=nnx.Rngs(keys[1]))
    outcome = None
    outcome_optimizer = None
    if explain.kind == "outcome":
        outcome = OutcomeNetworks(outcome_network(features, actions, explain.outcome), rngs=nnx.Rngs(keys[2]))
        outcome_optimizer = make_adam(outcome, explain.learning_rate, explain.adam)
    return Explainers(
        characteristic,
        make_adam(characteristic, explain.learning_rate, explain.adam),
        shapley,
        make_adam(shapley, explain.learning_rate, explain.adam),
        outcome,
        outcome_optimizer,
    )


def refresh_exact(
    state: ContinualState[object],
    *,
    env: EpisodeStatisticsWrapper[object, object],
    epsilon: Array,
    config: ContinualRunConfig[ContinualEnvironment],
) -> ContinualState[object]:
    """Source fill_buffer(dqn, copy(envs), exact_buffer, size, 'explore'): fresh explore rollouts."""
    keys: Array = jax.random.split(state.key)
    exact = collect_observations(
        env, state.run.model, epsilon, keys[1], config.explain.exact_buffer_size, config.train.num_envs
    )
    return ContinualState(state.run, state.explainers, exact.astype(jnp.int32), keys[0])


def agent_step(
    state: ContinualState[object],
    *,
    env: EpisodeStatisticsWrapper[object, object],
    params: object,
    training: DQNTrainConfig,
) -> tuple[ContinualState[object], StepMetrics]:
    run, metrics = training_step(state.run, env, params, training)
    return ContinualState(run, state.explainers, state.exact, state.key), metrics


def explanation_updates(
    state: ContinualState[object], *, epsilon: Array, config: ContinualRunConfig[ContinualEnvironment], features: int
) -> tuple[ContinualState[object], ExplanationLosses]:
    """`update_ratio` updates of each explanation model, in the source's model order.

    NNX graph nodes (the agent and the explanation models) travel in the loop carries:
    transforms accept them only at their own trace level.
    """
    explain, batch_size = config.explain, config.train.batch_size
    replay = state.run.replay
    explained = explain.characteristic
    keys: Array = jax.random.split(state.key, 4)
    type Carry = tuple[Explainers, QNetwork]

    def characteristic_update(carry: Carry, key: Array) -> tuple[Carry, Array]:
        explainers, agent = carry
        batch_key, coalition_key = jax.random.split(key)
        batch, _ = sample(replay, batch_size, batch_key)
        q_values = agent(batch.observations).q_values
        targets = explained_values(q_values, explained, epsilon)
        explore = action_log_probabilities(policy_probabilities(q_values, "explore", epsilon), batch.actions)
        weights = off_policy_weights(explore, batch.log_probabilities, explain.importance)
        coalitions = uniform_coalitions(coalition_key, batch_size, features)
        metrics = update_characteristic(
            explainers.characteristic,
            explainers.characteristic_optimizer,
            batch.observations,
            targets,
            coalitions,
            weights,
        )
        return carry, metrics.loss

    def outcome_update(carry: Carry, key: Array) -> tuple[Carry, Array]:
        explainers, agent = carry
        assert explainers.outcome is not None and explainers.outcome_optimizer is not None
        batch_key, explained_key, coalition_key, action_key = jax.random.split(key, 4)
        batch, _ = sample(replay, batch_size, batch_key)
        explained_rows, _ = sample(replay, batch_size, explained_key)
        coalitions = uniform_coalitions(coalition_key, batch_size, features)
        transitions = _transitions(batch)
        policy = conditioned_policy(explainers, agent, epsilon)
        next_actions = sample_actions(
            policy, action_key, transitions.next_observations, explained_rows.observations, coalitions
        )
        metrics = update_off_policy(
            explainers.outcome,
            explainers.outcome_optimizer,
            transitions,
            explained_rows.observations,
            coalitions,
            next_actions,
            explain.outcome,
        )
        return carry, metrics.loss

    def shapley_update(carry: Carry, key: Array) -> tuple[Carry, Array]:
        explainers, agent = carry
        batch_key, coalition_key = jax.random.split(key)
        batch, _ = sample(replay, batch_size, batch_key)
        coalition = kernel_coalitions(coalition_key, 1, features)[0]
        function = explained_function(explainers, agent, epsilon)
        metrics = update_shapley(
            explainers.shapley, explainers.shapley_optimizer, function, batch.observations, coalition
        )
        return carry, metrics.loss

    def repeat(update: Callable[[Carry, Array], tuple[Carry, Array]], carry: Carry, key: Array) -> tuple[Carry, Array]:
        carry, losses = nnx.scan(update, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            carry, jax.random.split(key, explain.update_ratio)
        )
        return carry, losses[-1]

    carry: Carry = (state.explainers, state.run.model)
    carry, characteristic_loss = repeat(characteristic_update, carry, keys[1])
    outcome_loss = jnp.asarray(nan, jnp.float32)
    if explain.kind == "outcome":
        carry, outcome_loss = repeat(outcome_update, carry, keys[2])
    carry, shapley_loss = repeat(shapley_update, carry, keys[3])
    losses = ExplanationLosses(characteristic_loss, outcome_loss, shapley_loss)
    return ContinualState(state.run, carry[0], state.exact, keys[0]), losses


def exact_losses(
    state: ContinualState[object], *, epsilon: Array, explain: ContinualConfig, tabular: TabularModel, max_unique: int
) -> ExactLosses:
    """Every model's exact loss against the current agent on the refreshed exact buffer."""
    explainers, agent = state.explainers, FrozenAgent(state.run.model, epsilon)
    observations = state.exact
    features = observations.shape[-1]
    coalitions = all_coalitions(features)
    targets = explained_values(agent.model(observations).q_values, explain.characteristic, epsilon)
    exact = exact_characteristic(observations, targets, coalitions, max_unique=max_unique)
    behaviour = MaskedCharacteristic(explainers.characteristic)
    characteristic = exact_characteristic_loss(explainers.characteristic, exact)
    if explain.kind != "outcome":
        values = (
            exact.at_states if explain.with_exact_char else characteristic_table(behaviour, exact.states, coalitions)
        )
        shapley = exact_shapley_loss(
            explainers.shapley, behaviour, exact.states, exact.state_valid, exact_shapley(values)
        )
        return ExactLosses(characteristic, jnp.asarray(nan, jnp.float32), shapley)
    assert explainers.outcome is not None
    policy = conditioned_policy(explainers, agent.model, epsilon)
    function = OutcomeCharacteristic(explainers.outcome, policy, "off_policy")
    outcome_values = outcome_exact(observations, policy, tabular, explain.outcome_with_exact_char, explain.gamma)
    outcome = exact_outcome_loss(function, outcome_values, coalitions)
    states = outcome_values.states
    explained = (
        outcome_values.values[..., None]
        if explain.with_exact_char
        else characteristic_table(function, states, coalitions)
    )
    shapley = exact_shapley_loss(
        explainers.shapley, function, states, outcome_values.state_valid, exact_shapley(explained)
    )
    return ExactLosses(characteristic, outcome, shapley)


def write_metrics(folder: Path, metrics: ContinualMetrics) -> None:
    """metrics.csv: logged and exact steps (the source's agent/* scalars); episodes.csv: completed episodes."""
    agent = metrics.agent
    steps = np.asarray(agent.global_step)
    logged, exact_step = np.asarray(metrics.logged), np.asarray(metrics.exact_step)
    columns = {
        "dqn_loss": agent.loss,
        "characteristic_loss": metrics.losses.characteristic,
        "outcome_loss": metrics.losses.outcome,
        "shapley_loss": metrics.losses.shapley,
        "characteristic_exact_loss": metrics.exact.characteristic,
        "outcome_exact_loss": metrics.exact.outcome,
        "shapley_exact_loss": metrics.exact.shapley,
    }
    values = {name: np.asarray(value, np.float64) for name, value in columns.items()}
    with (folder / "metrics.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("step", "epsilon", *columns))
        for row in np.flatnonzero(logged | exact_step):
            cells = [_cell(values[name][row], logged[row] or name.endswith("exact_loss")) for name in columns]
            writer.writerow((int(steps[row]), float(agent.epsilon[row]), *cells))
    completed = np.asarray(agent.completed)
    returns, lengths = np.asarray(agent.episode_return), np.asarray(agent.episode_length)
    with (folder / "episodes.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("step", "episode_return", "episode_length"))
        for row, column in zip(*np.nonzero(completed), strict=True):
            writer.writerow((int(steps[row]), float(returns[row, column]), int(lengths[row, column])))


def conditioned_policy(explainers: Explainers, agent: QNetwork, epsilon: Array) -> ConditionedPolicy:
    """pi_hat from the learnt behaviour characteristic and the agent's current explore policy."""
    return ConditionedPolicy(MaskedCharacteristic(explainers.characteristic), AgentPolicy(agent, "explore", epsilon))


def explained_function(explainers: Explainers, agent: QNetwork, epsilon: Array) -> CharacteristicFunction:
    """What the Shapley model explains: the learnt characteristic, or the learnt outcome characteristic."""
    if explainers.outcome is None:
        return MaskedCharacteristic(explainers.characteristic)
    return OutcomeCharacteristic(explainers.outcome, conditioned_policy(explainers, agent, epsilon), "off_policy")


def seed_slice(metrics: ContinualMetrics, index: int) -> ContinualMetrics:
    """One seed's metrics as host arrays, keeping the step axis."""
    return cast(ContinualMetrics, jax.tree.map(lambda value: np.asarray(value)[:, index], metrics))  # Same structure.


def _transitions(batch: TransitionBatch) -> Transitions:
    return Transitions(batch.observations, batch.actions, batch.rewards, batch.next_observations, batch.terminated)


def _no_losses(shape: tuple[int, ...]) -> ExplanationLosses:
    missing = jnp.full(shape, nan, jnp.float32)
    return ExplanationLosses(missing, missing, missing)


def _no_exact(shape: tuple[int, ...]) -> ExactLosses:
    missing = jnp.full(shape, nan, jnp.float32)
    return ExactLosses(missing, missing, missing)


def _cell(value: float, shown: bool) -> str:
    return "" if not shown or np.isnan(value) else repr(float(value))


def _settings(config: ContinualRunConfig[ContinualEnvironment]) -> dict[str, object]:
    return {"environment": type(config.env).__name__, "config": asdict(config)}


if __name__ == "__main__":
    main()
