"""FastSVERL explanations: configuration, compiled training and CLI.

A characteristic run explains a trained DQN agent (behaviour or prediction); an
outcome run explains the agent's return when it follows a trained behaviour
characteristic; a Shapley run explains a characteristic. Characteristic and
Shapley runs collect observation buffers with the agent; off-policy outcome runs
learn from prefixes of the agent's replay. All train for whole epochs with early
stopping and checkpoint complete state for exact resume. Shell recipes provide
experiment values; explanation_io.py owns host I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import IntEnum
from functools import partial
from math import isfinite, nan
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import tyro
from flax import nnx, struct
from jax import Array
from jax.experimental import io_callback

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    ExactCharacteristic,
    Kind,
    Policy,
    characteristic_loss,
    characteristic_network,
    exact_characteristic,
    exact_characteristic_loss,
    explained_values,
    update_characteristic,
)
from jax_shapley.coalitions import all_coalitions, kernel_coalitions, uniform_coalitions
from jax_shapley.envs import Gridworld, Mastermind222
from jax_shapley.networks import QNetwork
from jax_shapley.outcome import (
    AgentPolicy,
    ConditionedPolicy,
    ExactOutcome,
    OutcomeCharacteristic,
    OutcomeConfig,
    OutcomeNetworks,
    Transitions,
    Variant,
    exact_behaviour_table,
    exact_outcome,
    exact_outcome_loss,
    learnt_behaviour_table,
    off_policy_loss,
    outcome_network,
    sample_next_actions,
    update_off_policy,
)
from jax_shapley.replay import TransitionBatch
from jax_shapley.shapley import (
    CharacteristicFunction,
    MaskedCharacteristic,
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    shapley_loss,
    shapley_network,
    update_shapley,
)
from jax_shapley.tabular import TabularModel, transition_model
from jax_shapley.train_dqn import (
    AdamConfig,
    GridworldConfig,
    Mastermind222Config,
    ResumeConfig,
    choose_action,
    environment_step,
    make_adam,
)
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsState, EpisodeStatisticsWrapper

if TYPE_CHECKING:
    from jax_shapley.explanation_io import (
        ExplanationMetadata,
        ExplanationSink,
        ExplanationSnapshot,
        ExplanationSummary,
        RunContext,
    )

type BufferSource = Literal["fresh", "characteristic"]


class ExplanationPhase(IntEnum):
    TRAINING = 0
    COMPLETE = 1


@dataclass(frozen=True, slots=True)
class ExplanationTrainConfig:
    """Source train_models settings; buffer sizes count observations across all environments."""

    epochs: int = 50
    patience: int | None = None
    learning_rate: float = 2.5e-4
    batch_size: int = 128
    buffer_size: int = 10_000
    val_fraction: float = 0.2
    num_envs: int = 16
    exact_loss: bool = True
    exact_buffer_size: int | None = None
    exact_condition: float | None = None
    track_batch: bool = False
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("buffer_size", self.buffer_size),
            ("num_envs", self.num_envs),
        ):
            _positive(name, value)
        if self.patience is not None:
            _positive("patience", self.patience)
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        self._validate_buffers()
        if self.exact_condition is not None:
            if not self.exact_loss:
                raise ValueError("exact_condition requires exact_loss")
            if not isfinite(self.exact_condition) or self.exact_condition <= 0:
                raise ValueError("exact_condition must be positive")

    def _validate_buffers(self) -> None:
        sizes = [("train", self.train_size), ("validation", self.validation_size)]
        if self.exact_buffer_size is not None:
            sizes.append(("exact", self.exact_buffer_size))
        for name, size in sizes:
            if size < self.num_envs or size % self.num_envs:
                raise ValueError(f"{name} buffer size {size} must be a positive multiple of num_envs")
        if self.batches < 1:
            raise ValueError("the train buffer must hold at least one batch")

    @property
    def train_size(self) -> int:
        # Source arithmetic: int(buffer_size * (1 - val_fraction)).
        return int(self.buffer_size * (1 - self.val_fraction))

    @property
    def validation_size(self) -> int:
        return int(self.buffer_size * self.val_fraction)

    @property
    def exact_size(self) -> int:
        """Without a separate exact buffer, exact values use the train buffer."""
        return self.train_size if self.exact_buffer_size is None else self.exact_buffer_size

    @property
    def batches(self) -> int:
        return self.train_size // self.batch_size

    @property
    def patience_epochs(self) -> int:
        return self.epochs if self.patience is None else self.patience

    @property
    def checkpoint_epochs(self) -> int:
        return max(1, self.epochs // 10)


@dataclass(frozen=True, slots=True)
class CharacteristicRunConfig:
    agent: Path
    kind: Kind = "behaviour"
    policy: Policy = "greedy"
    hidden_widths: tuple[int, ...] = (64, 64)
    train: ExplanationTrainConfig = field(default_factory=ExplanationTrainConfig)
    seed: int = 1
    seeds: tuple[int, ...] = ()
    chunk_epochs: int = 1
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _run_options(self.seed, self.seeds, self.chunk_epochs, self.validate_only, self.compile_only)
        CharacteristicConfig(self.kind, self.policy, self.hidden_widths)

    @property
    def characteristic(self) -> CharacteristicConfig:
        return CharacteristicConfig(self.kind, self.policy, self.hidden_widths)

    @property
    def sweep(self) -> tuple[int, ...]:
        """Seeds trained together in one compiled program; a single run is a sweep of one."""
        return self.seeds or (self.seed,)

    def for_seed(self, seed: int, agent: Path | None = None) -> CharacteristicRunConfig:
        """The ordinary single-run settings each seed's folder records."""
        return replace(self, seed=seed, seeds=(), agent=agent or self.agent)


@dataclass(frozen=True, slots=True)
class ShapleyRunConfig:
    """Explains a completed characteristic run; kind and policy come from that run."""

    agent: Path
    characteristic: Path
    with_exact_char: bool = True
    buffers: BufferSource = "fresh"
    hidden_widths: tuple[int, ...] = (64, 64)
    train: ExplanationTrainConfig = field(default_factory=ExplanationTrainConfig)
    seed: int = 1
    seeds: tuple[int, ...] = ()
    chunk_epochs: int = 1
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _run_options(self.seed, self.seeds, self.chunk_epochs, self.validate_only, self.compile_only)
        if self.buffers not in ("fresh", "characteristic"):
            raise ValueError("buffers must be fresh or characteristic")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")

    @property
    def sweep(self) -> tuple[int, ...]:
        """Seeds trained together; `characteristic` may name a characteristic sweep, pairing seed s with its seed s."""
        return self.seeds or (self.seed,)

    def for_seed(self, seed: int, agent: Path | None = None, characteristic: Path | None = None) -> ShapleyRunConfig:
        return replace(
            self, seed=seed, seeds=(), agent=agent or self.agent, characteristic=characteristic or self.characteristic
        )


@dataclass(frozen=True, slots=True)
class OutcomeTrainConfig:
    """Source outcome settings: train_models over prefixes of the agent's replay (off-policy).

    The source's train and validation buffers are the first `train_size` and
    `validation_size` replay transitions, so they overlap.
    """

    epochs: int = 128
    patience: int | None = None
    learning_rate: float = 2.5e-4
    batch_size: int = 64
    train_size: int = 50_000
    validation_size: int = 80
    exact_loss: bool = True
    exact_condition: float | None = None
    track_batch: bool = True
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("train_size", self.train_size),
            ("validation_size", self.validation_size),
        ):
            _positive(name, value)
        if self.patience is not None:
            _positive("patience", self.patience)
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batches < 1:
            raise ValueError("the train buffer must hold at least one batch")
        if self.exact_condition is not None and (not self.exact_loss or not self.exact_condition > 0):
            raise ValueError("exact_condition must be positive and requires exact_loss")

    @property
    def batches(self) -> int:
        return self.train_size // self.batch_size

    @property
    def patience_epochs(self) -> int:
        return self.epochs if self.patience is None else self.patience

    @property
    def checkpoint_epochs(self) -> int:
        return max(1, self.epochs // 10)


@dataclass(frozen=True, slots=True)
class OutcomeRunConfig:
    """Explains the agent behind a completed behaviour characteristic run.

    `with_exact_char` selects the conditioned policy inside the exact ground truth:
    the exact or the learnt behaviour characteristic. Training always uses the
    learnt one.
    """

    agent: Path
    characteristic: Path
    variant: Variant = "off_policy"
    with_exact_char: bool = True
    tau: float = 0.001
    gamma: float = 1.0
    hidden_widths: tuple[int, ...] = (64, 64)
    train: OutcomeTrainConfig = field(default_factory=OutcomeTrainConfig)
    seed: int = 1
    seeds: tuple[int, ...] = ()
    chunk_epochs: int = 1
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _run_options(self.seed, self.seeds, self.chunk_epochs, self.validate_only, self.compile_only)
        OutcomeConfig(self.variant, self.tau, self.gamma, self.hidden_widths)
        if self.variant != "off_policy":
            raise ValueError("only the off-policy outcome characteristic is available")

    @property
    def outcome(self) -> OutcomeConfig:
        return OutcomeConfig(self.variant, self.tau, self.gamma, self.hidden_widths)

    @property
    def sweep(self) -> tuple[int, ...]:
        """Seeds trained together; `characteristic` may name a characteristic sweep, pairing seed s with its seed s."""
        return self.seeds or (self.seed,)

    def for_seed(self, seed: int, agent: Path | None = None, characteristic: Path | None = None) -> OutcomeRunConfig:
        return replace(
            self, seed=seed, seeds=(), agent=agent or self.agent, characteristic=characteristic or self.characteristic
        )


type RunConfig = CharacteristicRunConfig | ShapleyRunConfig | OutcomeRunConfig
type TrainConfig = ExplanationTrainConfig | OutcomeTrainConfig
type ExplainedModel = CharacteristicModel | ShapleyModel | OutcomeNetworks
type AnyState = (
    ExplanationState[CharacteristicModel] | ExplanationState[ShapleyModel] | ExplanationState[OutcomeNetworks]
)


@struct.dataclass
class Buffers:
    """int32 observations recorded before each environment step."""

    train: Array
    validation: Array
    exact: Array


@struct.dataclass
class ExplanationState[Model]:
    model: Model
    best: Model
    optimizer: nnx.Optimizer[Model]
    buffers: Buffers
    epoch: Array
    best_loss: Array
    best_epoch: Array
    stopped: Array
    phase: Array
    key: Array


@struct.dataclass
class EpochMetrics:
    epoch: Array
    step: Array
    train_loss: Array
    validation_loss: Array
    exact_loss: Array
    best_epoch: Array
    active: Array


@struct.dataclass
class FrozenAgent:
    """The explained agent: Q-network and the epsilon its explore policy uses."""

    model: QNetwork
    epsilon: Array


@struct.dataclass
class CharacteristicContext:
    """Per-seed constants a characteristic learner reads: buffers, targets and exact values."""

    buffers: Buffers
    train_targets: Array
    validation_targets: Array
    exact: ExactCharacteristic | None


@struct.dataclass
class ShapleyContext:
    """Per-seed constants a Shapley learner reads: buffers and the exact-loss ground truth."""

    buffers: Buffers
    exact: ExactCharacteristic | None
    exact_values: Array | None


@struct.dataclass
class OutcomeInputs:
    """Per-seed inputs of an outcome run: the behaviour characteristic's buffers and the agent's replay prefixes."""

    buffers: Buffers
    train: Transitions
    validation: Transitions


@struct.dataclass
class OutcomeFrozen:
    """The frozen models an outcome learner conditions on: the agent and its behaviour characteristic."""

    agent: FrozenAgent
    behaviour: CharacteristicModel


@struct.dataclass
class OutcomeContext:
    """Per-seed constants an outcome learner reads: replay prefixes and the exact ground truth."""

    train: Transitions
    validation: Transitions
    exact: ExactOutcome | None


class Draw(NamedTuple):
    """One batch's sampled buffer rows and coalition(s)."""

    rows: Array
    coalitions: Array


class OutcomeDraw(NamedTuple):
    """One off-policy outcome batch: transition rows, independently drawn explained rows, coalitions per row,
    and the key the conditioned policy samples next actions with."""

    rows: Array
    explained: Array
    coalitions: Array
    key: Array


class Schedule(Protocol):
    """Epoch-loop settings shared by every train_models-style run."""

    @property
    def epochs(self) -> int: ...
    @property
    def learning_rate(self) -> float: ...
    @property
    def exact_loss(self) -> bool: ...
    @property
    def exact_condition(self) -> float | None: ...
    @property
    def track_batch(self) -> bool: ...
    @property
    def adam(self) -> AdamConfig: ...
    @property
    def batches(self) -> int: ...
    @property
    def patience_epochs(self) -> int: ...
    @property
    def checkpoint_epochs(self) -> int: ...


class Learner[Model, Sample](NamedTuple):
    """Model-specific steps closed over precomputed targets and frozen models.

    Sampling is separate from learning: `train_draw(key, epoch, batch)` and
    `validation_draw(key, epoch)` default to the source's random draws, and
    parity tests replay recorded upstream draws through them instead.
    """

    train_batch: Callable[[Model, nnx.Optimizer[Model], Buffers, Sample], Array]
    validation_loss: Callable[[Model, Buffers, Sample], Array]
    exact_loss: Callable[[Model], Array]
    train_draw: Callable[[Array, Array, Array], Sample]
    validation_draw: Callable[[Array, Array], Sample]


class Runner(Protocol):
    def __call__(self, *args: object) -> AnyState: ...


type CLIConfig = (
    Annotated[CharacteristicRunConfig, tyro.conf.subcommand(name="characteristic")]
    | Annotated[ShapleyRunConfig, tyro.conf.subcommand(name="shapley")]
    | Annotated[OutcomeRunConfig, tyro.conf.subcommand(name="outcome")]
    | Annotated[ResumeConfig, tyro.conf.subcommand(name="resume")]
)


def main(args: list[str] | None = None) -> ExplanationSummary | list[ExplanationSummary] | None:
    """Validate, restore, compile, execute once and finalize host artifacts.

    A run with `--seeds` trains every seed in one compiled program and returns one
    summary per seed; each seed's folder is an ordinary run folder.
    """
    from contextlib import ExitStack

    from jax_shapley.explanation_io import (
        CallbackRouter,
        ExplanationWriter,
        SweepWriter,
        create_run,
        create_sweep,
        describe_options,
        load_run_settings,
        prepare_context,
        summarize,
        sweep_folders,
        write_summary,
    )

    options = parse_args(args)
    resume = options if isinstance(options, ResumeConfig) else None
    if resume is not None:
        folders: list[Path] | None = sweep_folders(resume.run)
        configs = [load_run_settings(folder / "config.json").config for folder in folders]
        config = configs[0]
    else:
        assert not isinstance(options, ResumeConfig)
        config, folders = options, None
        if config.validate_only:
            print(describe_options(config), end="")
            return None
        configs = seed_configs(config)
    contexts = [prepare_context(seed_config) for seed_config in configs]
    restored = None
    metadata: list[ExplanationMetadata | None] = [None] * len(configs)
    sequences: list[int | None] = [None] * len(configs)
    if resume is not None and folders is not None:
        restored, metadata, sequences = restore_sweep(folders, configs, contexts)
        if resume.validate_only:
            print("".join(context.settings for context in contexts), end="")
            return None
    router = CallbackRouter()
    runner = nnx.jit(_bind(contexts[0], configs[0], router))
    keys = jnp.stack([jax.random.key(seed_config.seed) for seed_config in configs])
    agents = stack_seeds([context.agent for context in contexts])
    characteristics = _stack_optional([context.characteristic for context in contexts])
    inherited = _stack_optional([context.inherited for context in contexts])
    abstract = nnx.eval_shape(lambda state: state, restored) if restored is not None else None
    key_spec = jax.ShapeDtypeStruct(keys.shape, keys.dtype)
    compiled = cast(Runner, runner.trace(key_spec, agents, characteristics, inherited, abstract).lower().compile())
    if config.compile_only or (resume is not None and resume.compile_only):
        print(
            f"Compiled {contexts[0].environment} {_stage(config)} on {jax.default_backend()}; no run artifacts created."
        )
        return None
    if folders is None:
        folders = [create_run(configs[0], contexts[0])] if not config.seeds else create_sweep(config, contexts)
    with ExitStack() as stack:
        writers = [
            stack.enter_context(
                ExplanationWriter(folder, context.settings, metadata=saved, checkpoint_sequence=sequence)
            )
            for folder, context, saved, sequence in zip(folders, contexts, metadata, sequences, strict=True)
        ]
        sink = SweepWriter(writers)
        router.sink = sink
        try:
            result = compiled(keys, agents, characteristics, inherited, restored)
            jax.block_until_ready(result.epoch)
            jax.effects_barrier()
            sink.wait()
        finally:
            router.sink = None
        summaries = []
        for index, (folder, seed_config, writer) in enumerate(zip(folders, configs, writers, strict=True)):
            summary = summarize(folder, seed_config, seed_slice(result, index), writer.elapsed())
            write_summary(folder, summary)
            summaries.append(summary)
    for summary in summaries:
        print(
            f"Trained {contexts[0].environment} {_stage(config)}: {summary.epochs} epochs. Artifacts: {summary.folder}"
        )
    return summaries[0] if len(summaries) == 1 and not config.seeds else summaries


def restore_sweep(
    folders: list[Path], configs: list[RunConfig], contexts: list[RunContext]
) -> tuple[
    AnyState,
    list[ExplanationMetadata | None],
    list[int | None],
]:
    """Restore each seed folder's checkpoint, check its explained models, and stack the states."""
    from jax_shapley.explanation_io import ExplanationStore, load_run_settings, restore_state

    states: list[AnyState] = []
    metadata: list[ExplanationMetadata | None] = []
    sequences: list[int | None] = []
    for folder, seed_config, context in zip(folders, configs, contexts, strict=True):
        saved = load_run_settings(folder / "config.json")
        if (saved.agent_sha256, saved.characteristic_sha256) != (context.agent_sha256, context.characteristic_sha256):
            raise ValueError("explained agent or characteristic changed since the run was created")
        store = ExplanationStore(folder, read_only=True)
        try:
            state, saved_metadata = restore_state(store, template_state(context, seed_config), seed_config, context)
            states.append(state)
            metadata.append(saved_metadata)
            sequences.append(store.sequence)
        finally:
            store.close()
    return stack_seeds(states), metadata, sequences


def seed_configs(config: RunConfig) -> list[RunConfig]:
    """Each seed's single-run settings.

    `agent` may name one DQN run (shared by every seed) or a directory of DQN runs,
    paired by their training seed. A Shapley `characteristic` may name one run or a
    characteristic sweep, pairing seed s with its seed s.
    """
    from jax_shapley.explanation_io import sweep_folders
    from jax_shapley.run_io import load_settings

    agents: dict[int, Path] | None = None
    if not (config.agent / "config.json").exists():
        runs = [folder for folder in sorted(config.agent.iterdir()) if (folder / "config.json").exists()]
        agents = {load_settings(folder / "config.json").seed: folder for folder in runs}
    characteristics: dict[int, Path] | None = None
    if not isinstance(config, CharacteristicRunConfig):
        paired = sweep_folders(config.characteristic)
        if paired != [config.characteristic]:
            characteristics = {int(folder.name.removeprefix("seed-")): folder for folder in paired}
    for name, pairs in (("agent", agents), ("characteristic", characteristics)):
        missing = [seed for seed in config.sweep if pairs is not None and seed not in pairs]
        if missing:
            raise ValueError(f"no {name} for seeds {missing}")
    if isinstance(config, CharacteristicRunConfig):
        return [config.for_seed(seed, agents[seed] if agents else None) for seed in config.sweep]
    return [
        config.for_seed(seed, agents[seed] if agents else None, characteristics[seed] if characteristics else None)
        for seed in config.sweep
    ]


def stack_seeds[Tree](items: list[Tree]) -> Tree:
    """Stack equally shaped pytrees (NNX modules included) along a new leading seed axis."""
    graph, first = nnx.split(items[0])
    states = [first, *(nnx.split(item)[1] for item in items[1:])]
    return nnx.merge(graph, jax.tree.map(lambda *leaves: jnp.stack(leaves), *states))


def seed_slice[Tree](tree: Tree, seed: int) -> Tree:
    """One seed of a seed-stacked pytree."""
    graph, state = nnx.split(tree)
    return nnx.merge(graph, jax.tree.map(lambda leaf: leaf[seed], state))


def run_characteristic[State, Params](
    keys: Array,
    agents: FrozenAgent,
    characteristics: CharacteristicModel | None,
    inherited: Buffers | None,
    state: ExplanationState[CharacteristicModel] | None = None,
    *,
    environment_factory: Callable[[], EnvironmentProtocol[State, Params]],
    config: CharacteristicRunConfig,
    max_unique: int,
    sink: ExplanationSink,
) -> ExplanationState[CharacteristicModel]:
    """Collect buffers (fresh runs), then train every seed's characteristic to completion.

    Inputs and state carry a leading seed axis; each seed uses exactly the keys a
    single run with that seed would.
    """
    del characteristics, inherited
    env = EpisodeStatisticsWrapper(environment_factory())
    explained = config.characteristic
    training = config.train
    features, actions = _dimensions(env)
    network = characteristic_network(features, actions, explained)
    if state is None:

        def initialize(key: Array, agent: FrozenAgent) -> ExplanationState[CharacteristicModel]:
            key, buffers = collect_buffers(env, agent, explained.policy, training, key, None)
            model_key, key = jax.random.split(key)
            return initialize_state(CharacteristicModel(network, rngs=nnx.Rngs(model_key)), buffers, training, key)

        state = nnx.vmap(initialize, in_axes=(0, 0), out_axes=0)(keys, agents)

    def context(buffers: Buffers, agent: FrozenAgent) -> CharacteristicContext:
        return characteristic_context(buffers, agent, explained, training, features, max_unique)

    contexts = nnx.vmap(context, in_axes=(0, 0), out_axes=0)(state.buffers, agents)

    def learner(context: CharacteristicContext, frozen: None) -> Learner[CharacteristicModel, Draw]:
        del frozen
        return characteristic_learner(context, training, features)

    return train(state, contexts, None, learner, training, config.chunk_epochs, sink)


def run_shapley[State, Params](
    keys: Array,
    agents: FrozenAgent,
    characteristics: CharacteristicModel | None,
    inherited: Buffers | None,
    state: ExplanationState[ShapleyModel] | None = None,
    *,
    environment_factory: Callable[[], EnvironmentProtocol[State, Params]],
    config: ShapleyRunConfig,
    explained: CharacteristicConfig,
    max_unique: int,
    sink: ExplanationSink,
) -> ExplanationState[ShapleyModel]:
    """Explain each seed's frozen characteristic; exact values always use its exact buffer."""
    if characteristics is None or inherited is None:
        raise ValueError("Shapley runs need the trained characteristics and their buffers")
    env = EpisodeStatisticsWrapper(environment_factory())
    training = config.train
    features, actions = _dimensions(env)
    layout = ShapleyLayout(features, actions if explained.kind == "behaviour" else 1)
    network = shapley_network(layout, config.hidden_widths)
    if state is None:

        def initialize(key: Array, agent: FrozenAgent, inherited: Buffers) -> ExplanationState[ShapleyModel]:
            if config.buffers == "characteristic":
                buffers = inherited
            else:
                key, buffers = collect_buffers(env, agent, explained.policy, training, key, inherited.exact)
            model_key, key = jax.random.split(key)
            return initialize_state(ShapleyModel(network, layout, rngs=nnx.Rngs(model_key)), buffers, training, key)

        state = nnx.vmap(initialize, in_axes=(0, 0, 0), out_axes=0)(keys, agents, inherited)

    def context(buffers: Buffers, agent: FrozenAgent, characteristic: CharacteristicModel) -> ShapleyContext:
        function = MaskedCharacteristic(characteristic)
        return shapley_context(buffers, agent, function, explained, config, features, max_unique)

    contexts = nnx.vmap(context, in_axes=(0, 0, 0), out_axes=0)(state.buffers, agents, characteristics)

    def learner(context: ShapleyContext, characteristic: CharacteristicModel) -> Learner[ShapleyModel, Draw]:
        return shapley_learner(context, MaskedCharacteristic(characteristic), config, features)

    return train(state, contexts, characteristics, learner, training, config.chunk_epochs, sink)


def run_outcome[State, Params](
    keys: Array,
    agents: FrozenAgent,
    characteristics: CharacteristicModel | None,
    inherited: OutcomeInputs | None,
    state: ExplanationState[OutcomeNetworks] | None = None,
    *,
    environment_factory: Callable[[], EnvironmentProtocol[State, Params]],
    config: OutcomeRunConfig,
    explained: CharacteristicConfig,
    tabular: TabularModel | None,
    sink: ExplanationSink,
) -> ExplanationState[OutcomeNetworks]:
    """Learn each seed's off-policy outcome characteristic from its agent's replay prefixes.

    The buffers checkpointed with the state are the replay prefixes' observations
    and the behaviour characteristic's exact buffer, whose unique states the exact
    values explain.
    """
    if characteristics is None or inherited is None:
        raise ValueError("outcome runs need the behaviour characteristics, their buffers and the agents' replay")
    env = EpisodeStatisticsWrapper(environment_factory())
    training = config.train
    features, actions = _dimensions(env)
    network = outcome_network(features, actions, config.outcome)
    if state is None:

        def initialize(key: Array, inputs: OutcomeInputs) -> ExplanationState[OutcomeNetworks]:
            model_key, key = jax.random.split(key)
            buffers = Buffers(inputs.train.observations, inputs.validation.observations, inputs.buffers.exact)
            return initialize_state(OutcomeNetworks(network, rngs=nnx.Rngs(model_key)), buffers, training, key)

        state = nnx.vmap(initialize, in_axes=(0, 0), out_axes=0)(keys, inherited)

    def context(inputs: OutcomeInputs, agent: FrozenAgent, behaviour: CharacteristicModel) -> OutcomeContext:
        return outcome_context(inputs, OutcomeFrozen(agent, behaviour), explained, config, tabular)

    contexts = nnx.vmap(context, in_axes=(0, 0, 0), out_axes=0)(inherited, agents, characteristics)
    frozen = OutcomeFrozen(agents, characteristics)

    def learner(context: OutcomeContext, frozen: OutcomeFrozen) -> Learner[OutcomeNetworks, OutcomeDraw]:
        return outcome_learner(context, frozen, explained, config, features)

    return train(state, contexts, frozen, learner, training, config.chunk_epochs, sink)


def outcome_context(
    inputs: OutcomeInputs,
    frozen: OutcomeFrozen,
    explained: CharacteristicConfig,
    config: OutcomeRunConfig,
    tabular: TabularModel | None,
) -> OutcomeContext:
    """Exact outcome values at the behaviour characteristic's unique exact-buffer states.

    Those states are non-terminal, so they are exactly the transition model's
    states that occur in the exact buffer. with_exact_char conditions on the exact
    behaviour characteristic, otherwise on the learnt one; both follow the agent's
    explained policy elsewhere.
    """
    if not config.train.exact_loss:
        return OutcomeContext(inputs.train, inputs.validation, None)
    if tabular is None:
        raise ValueError("exact outcome values need the environment's transition model")
    policy = conditioned_policy(frozen, explained)
    observations, states = inputs.buffers.exact, tabular.states
    coalitions = all_coalitions(states.shape[-1])
    exact_table, present = exact_behaviour_table(observations, policy.agent(observations), states, coalitions)
    conditioned = (
        exact_table if config.with_exact_char else learnt_behaviour_table(policy.behaviour, states, coalitions)
    )
    exact = exact_outcome(tabular, states, present, conditioned, policy.agent(states), config.gamma)
    return OutcomeContext(inputs.train, inputs.validation, exact)


def outcome_learner(
    context: OutcomeContext,
    frozen: OutcomeFrozen,
    explained: CharacteristicConfig,
    config: OutcomeRunConfig,
    features: int,
) -> Learner[OutcomeNetworks, OutcomeDraw]:
    """Source get_data for PerformanceCharacteristicOffPolicy: transitions plus an independent e_obs sample."""
    training = config.train
    outcome = config.outcome
    policy = conditioned_policy(frozen, explained)

    def draw(key: Array, size: int, buffer: int) -> OutcomeDraw:
        keys: Array = jax.random.split(key, 4)
        rows = jax.random.randint(keys[0], (size,), 0, buffer)
        explained_rows = jax.random.randint(keys[1], (size,), 0, buffer)
        return OutcomeDraw(rows, explained_rows, uniform_coalitions(keys[2], size, features), keys[3])

    def train_draw(key: Array, epoch: Array, batch: Array) -> OutcomeDraw:
        del epoch, batch
        return draw(key, training.batch_size, training.train_size)

    def validation_draw(key: Array, epoch: Array) -> OutcomeDraw:
        del epoch
        # Source validation resamples the whole validation size with replacement.
        return draw(key, training.validation_size, training.validation_size)

    def inputs(transitions: Transitions, sample: OutcomeDraw) -> tuple[Transitions, Array, Array]:
        rows = sample.rows
        batch = Transitions(
            transitions.observations[rows],
            transitions.actions[rows],
            transitions.rewards[rows],
            transitions.next_observations[rows],
            transitions.terminated[rows],
        )
        explained_rows = transitions.observations[sample.explained]
        next_actions = sample_next_actions(
            policy, sample.key, batch.next_observations, explained_rows, sample.coalitions
        )
        return batch, explained_rows, next_actions

    def train_batch(
        networks: OutcomeNetworks, optimizer: nnx.Optimizer[OutcomeNetworks], buffers: Buffers, sample: OutcomeDraw
    ) -> Array:
        del buffers
        batch, explained_rows, next_actions = inputs(context.train, sample)
        return update_off_policy(
            networks, optimizer, batch, explained_rows, sample.coalitions, next_actions, outcome
        ).loss

    def validation_loss(networks: OutcomeNetworks, buffers: Buffers, sample: OutcomeDraw) -> Array:
        del buffers
        batch, explained_rows, next_actions = inputs(context.validation, sample)
        return off_policy_loss(
            networks.model, networks.target, batch, explained_rows, sample.coalitions, next_actions, outcome.gamma
        )[0]

    def exact_loss(networks: OutcomeNetworks) -> Array:
        assert context.exact is not None
        characteristic = OutcomeCharacteristic(networks, policy, outcome.variant)
        return exact_outcome_loss(characteristic, context.exact, all_coalitions(features))

    return Learner(train_batch, validation_loss, exact_loss, train_draw, validation_draw)


def conditioned_policy(frozen: OutcomeFrozen, explained: CharacteristicConfig) -> ConditionedPolicy:
    """pi_hat from the behaviour characteristic and the agent's explained (greedy or explore) policy."""
    agent = AgentPolicy(frozen.agent.model, explained.policy, frozen.agent.epsilon)
    return ConditionedPolicy(MaskedCharacteristic(frozen.behaviour), agent)


def replay_prefix(data: TransitionBatch, rows: int) -> Transitions:
    """The first `rows` replay transitions, as the source slices `agent.buffer` for outcome buffers."""
    if data.observations.shape[0] < rows:
        raise ValueError(f"the agent's replay holds fewer than {rows} transitions")
    return Transitions(
        data.observations[:rows],
        data.actions[:rows],
        data.rewards[:rows],
        data.next_observations[:rows],
        data.terminated[:rows],
    )


def characteristic_context(
    buffers: Buffers,
    agent: FrozenAgent,
    explained: CharacteristicConfig,
    training: ExplanationTrainConfig,
    features: int,
    max_unique: int,
) -> CharacteristicContext:
    """Per-run constants: the frozen agent's explained outputs and the exact values on the exact buffer."""

    def targets(observations: Array) -> Array:
        return explained_values(agent.model(observations).q_values, explained, agent.epsilon)

    exact = (
        exact_characteristic(buffers.exact, targets(buffers.exact), all_coalitions(features), max_unique=max_unique)
        if training.exact_loss
        else None
    )
    return CharacteristicContext(buffers, targets(buffers.train), targets(buffers.validation), exact)


def characteristic_learner(
    context: CharacteristicContext, training: ExplanationTrainConfig, features: int
) -> Learner[CharacteristicModel, Draw]:
    """Uniform coalitions per row; targets and exact values come from the precomputed context."""

    def train_draw(key: Array, epoch: Array, batch: Array) -> Draw:
        del epoch, batch
        index_key, coalition_key = jax.random.split(key)
        rows = jax.random.randint(index_key, (training.batch_size,), 0, training.train_size)
        return Draw(rows, uniform_coalitions(coalition_key, training.batch_size, features))

    def validation_draw(key: Array, epoch: Array) -> Draw:
        del epoch
        index_key, coalition_key = jax.random.split(key)
        # Source validation resamples the whole validation size with replacement.
        rows = jax.random.randint(index_key, (training.validation_size,), 0, training.validation_size)
        return Draw(rows, uniform_coalitions(coalition_key, training.validation_size, features))

    def train_batch(
        model: CharacteristicModel, optimizer: nnx.Optimizer[CharacteristicModel], buffers: Buffers, draw: Draw
    ) -> Array:
        rows = draw.rows
        targets = context.train_targets[rows]
        return update_characteristic(model, optimizer, buffers.train[rows], targets, draw.coalitions).loss

    def validation_loss(model: CharacteristicModel, buffers: Buffers, draw: Draw) -> Array:
        rows = draw.rows
        targets = context.validation_targets[rows]
        return characteristic_loss(model, buffers.validation[rows], targets, draw.coalitions)[0]

    def exact_loss(model: CharacteristicModel) -> Array:
        assert context.exact is not None
        return exact_characteristic_loss(model, context.exact)

    return Learner(train_batch, validation_loss, exact_loss, train_draw, validation_draw)


def shapley_context(
    buffers: Buffers,
    agent: FrozenAgent,
    characteristic: CharacteristicFunction,
    explained: CharacteristicConfig,
    config: ShapleyRunConfig,
    features: int,
    max_unique: int,
) -> ShapleyContext:
    """Exact Shapley values on the characteristic's exact buffer; with_exact_char selects the ground truth."""
    if not config.train.exact_loss:
        return ShapleyContext(buffers, None, None)
    observations = buffers.exact
    targets = explained_values(agent.model(observations).q_values, explained, agent.epsilon)
    coalitions = all_coalitions(features)
    exact = exact_characteristic(observations, targets, coalitions, max_unique=max_unique)
    # Training always uses the learnt characteristic; with_exact_char only changes the ground truth.
    values = (
        exact.at_states if config.with_exact_char else characteristic_table(characteristic, exact.states, coalitions)
    )
    return ShapleyContext(buffers, exact, exact_shapley(values))


def shapley_learner(
    context: ShapleyContext, characteristic: CharacteristicFunction, config: ShapleyRunConfig, features: int
) -> Learner[ShapleyModel, Draw]:
    """One Shapley-kernel coalition per batch against the frozen characteristic."""
    training = config.train

    def train_draw(key: Array, epoch: Array, batch: Array) -> Draw:
        del epoch, batch
        index_key, coalition_key = jax.random.split(key)
        rows = jax.random.randint(index_key, (training.batch_size,), 0, training.train_size)
        return Draw(rows, kernel_coalitions(coalition_key, 1, features)[0])

    def validation_draw(key: Array, epoch: Array) -> Draw:
        del epoch
        index_key, coalition_key = jax.random.split(key)
        rows = jax.random.randint(index_key, (training.validation_size,), 0, training.validation_size)
        return Draw(rows, kernel_coalitions(coalition_key, 1, features)[0])

    def train_batch(model: ShapleyModel, optimizer: nnx.Optimizer[ShapleyModel], buffers: Buffers, draw: Draw) -> Array:
        return update_shapley(model, optimizer, characteristic, buffers.train[draw.rows], draw.coalitions).loss

    def validation_loss(model: ShapleyModel, buffers: Buffers, draw: Draw) -> Array:
        return shapley_loss(model, characteristic, buffers.validation[draw.rows], draw.coalitions)[0]

    def exact_loss(model: ShapleyModel) -> Array:
        exact = context.exact
        assert exact is not None and context.exact_values is not None
        return exact_shapley_loss(model, characteristic, exact.states, exact.state_valid, context.exact_values)

    return Learner(train_batch, validation_loss, exact_loss, train_draw, validation_draw)


def train[Model: nnx.Module, Context, Frozen, Sample](
    state: ExplanationState[Model],
    contexts: Context,
    frozen: Frozen,
    make_learner: Callable[[Context, Frozen], Learner[Model, Sample]],
    config: Schedule,
    chunk_epochs: int,
    sink: ExplanationSink,
) -> ExplanationState[Model]:
    """All seeds' epochs in compiled blocks: metrics flush per block, checkpoints every tenth of the epochs.

    `state`, `contexts` and `frozen` carry a leading seed axis (`frozen` may be None).
    Finished seeds no-op while others continue; the loop ends when every seed is done.
    """

    def running(state: ExplanationState[Model]) -> Array:
        return (state.phase == ExplanationPhase.TRAINING) & (state.epoch < config.epochs) & ~state.stopped

    def seed_step(
        state: ExplanationState[Model], context: Context, frozen: Frozen
    ) -> tuple[ExplanationState[Model], EpochMetrics]:
        learner = make_learner(context, frozen)
        result = nnx.cond(
            running(state),
            lambda value: epoch_step(value, learner, config),
            lambda value: (value, _inactive_metrics()),
            state,
        )
        return cast(tuple[ExplanationState[Model], EpochMetrics], result)  # NNX cond keeps the carry topology.

    every_seed = nnx.vmap(seed_step, in_axes=(0, 0, None if frozen is None else 0), out_axes=(0, 0))

    # Frozen models are graph nodes: NNX transforms only accept them at their own trace level,
    # so they travel in the loop carries rather than being captured from the outer trace.
    type Carry = tuple[ExplanationState[Model], Frozen]

    def block(carry: Carry) -> Carry:
        state, frozen = carry
        start = state.epoch

        def step(carry: Carry, index: Array) -> tuple[Carry, EpochMetrics]:
            del index
            state, frozen = carry
            state, metrics = every_seed(state, contexts, frozen)
            return (state, frozen), metrics

        (state, frozen), metrics = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            (state, frozen), jnp.arange(chunk_epochs, dtype=jnp.int32)
        )
        _flush(metrics, sink)
        due = (state.epoch // config.checkpoint_epochs > start // config.checkpoint_epochs) & running(state)
        jax.lax.cond(jnp.any(due), lambda: _emit_checkpoint(state, due, sink), lambda: jnp.asarray(0, jnp.int32))
        return state, frozen

    trained, _ = nnx.while_loop(lambda carry: jnp.any(running(carry[0])), block, (state, frozen))
    finishing = trained.phase == ExplanationPhase.TRAINING

    def finish(state: ExplanationState[Model]) -> ExplanationState[Model]:
        result = nnx.cond(state.phase == ExplanationPhase.TRAINING, finalize, lambda value: value, state)
        return cast(ExplanationState[Model], result)  # NNX cond keeps the carry topology.

    finished = nnx.vmap(finish, in_axes=0, out_axes=0)(trained)
    jax.lax.cond(
        jnp.any(finishing),
        lambda: _emit_checkpoint(finished, finishing, sink),
        lambda: jnp.asarray(0, jnp.int32),
    )
    return finished


def epoch_step[Model: nnx.Module, Sample](
    state: ExplanationState[Model], learner: Learner[Model, Sample], config: Schedule
) -> tuple[ExplanationState[Model], EpochMetrics]:
    """One source epoch: shuffled-with-replacement batches, validation, exact loss, early stopping."""
    key, batch_key, validation_key = jax.random.split(state.key, 3)

    def batch(
        carry: tuple[Model, nnx.Optimizer[Model]], inputs: tuple[Array, Array]
    ) -> tuple[tuple[Model, nnx.Optimizer[Model]], Array]:
        model, optimizer = carry
        key, index = inputs
        draw = learner.train_draw(key, state.epoch, index)
        return (model, optimizer), learner.train_batch(model, optimizer, state.buffers, draw)

    batches = (jax.random.split(batch_key, config.batches), jnp.arange(config.batches, dtype=jnp.int32))
    _, losses = nnx.scan(batch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
        (state.model, state.optimizer), batches
    )
    validation = learner.validation_loss(
        state.model, state.buffers, learner.validation_draw(validation_key, state.epoch)
    )
    exact = learner.exact_loss(state.model) if config.exact_loss else jnp.asarray(nan, jnp.float32)
    epoch = state.epoch + 1
    improved = validation < state.best_loss
    nnx.cond(improved, _copy_parameters, lambda source, destination: None, state.model, state.best)
    best_loss = jnp.where(improved, validation, state.best_loss)
    best_epoch = jnp.where(improved, epoch, state.best_epoch)
    stopped = epoch - best_epoch >= config.patience_epochs
    if config.exact_condition is not None:
        stopped = stopped | (exact < config.exact_condition)
    # Source logging step: the epoch, or (epoch + 1) * batches under track_batch.
    step = (epoch + 1) * config.batches if config.track_batch else epoch
    next_state = ExplanationState(
        state.model,
        state.best,
        state.optimizer,
        state.buffers,
        epoch,
        best_loss,
        best_epoch,
        stopped,
        state.phase,
        key,
    )
    metrics = EpochMetrics(epoch, step, jnp.mean(losses), validation, exact, best_epoch, jnp.asarray(True))
    return next_state, metrics


def finalize[Model: nnx.Module](state: ExplanationState[Model]) -> ExplanationState[Model]:
    """Restore the best-validation model and mark completion; the caller commits the checkpoint."""
    _copy_parameters(state.best, state.model)
    return ExplanationState(
        state.model,
        state.best,
        state.optimizer,
        state.buffers,
        state.epoch,
        state.best_loss,
        state.best_epoch,
        state.stopped,
        jnp.asarray(ExplanationPhase.COMPLETE, jnp.int32),
        state.key,
    )


def collect_buffers[State, Params](
    env: EpisodeStatisticsWrapper[State, Params],
    agent: FrozenAgent,
    policy: Policy,
    config: ExplanationTrainConfig,
    key: Array,
    exact: Array | None,
) -> tuple[Array, Buffers]:
    """Source fill_buffer for train, validation and (optionally) a separate exact buffer.

    Each buffer starts from a fresh reset; without a separate exact buffer the
    train buffer is used, unless `exact` is supplied by an earlier run.
    """
    epsilon = agent.epsilon if policy == "explore" else jnp.asarray(0.0, jnp.float32)
    keys: Array = jax.random.split(key, 4)
    key, train_key, validation_key, exact_key = keys[0], keys[1], keys[2], keys[3]
    train_rows = collect_observations(env, agent.model, epsilon, train_key, config.train_size, config.num_envs)
    validation = collect_observations(
        env, agent.model, epsilon, validation_key, config.validation_size, config.num_envs
    )
    if exact is None:
        exact = (
            train_rows
            if config.exact_buffer_size is None
            else collect_observations(env, agent.model, epsilon, exact_key, config.exact_buffer_size, config.num_envs)
        )
    return key, Buffers(train_rows, validation, exact)


def collect_observations[State, Params](
    env: EpisodeStatisticsWrapper[State, Params],
    model: QNetwork,
    epsilon: Array,
    key: Array,
    rows: int,
    num_envs: int,
) -> Array:
    """Observations before each vector step, step-major then environment, with Gymnax auto-reset."""
    params = env.default_params
    keys: Array = jax.random.split(key, num_envs + 1)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys[1:], params)

    def step(
        carry: tuple[Array, EpisodeStatisticsState[State], Array], index: Array
    ) -> tuple[tuple[Array, EpisodeStatisticsState[State], Array], Array]:
        del index
        current, state, key = carry
        actions, key = choose_action(model, current, epsilon, key)
        step_keys: Array = jax.random.split(key, num_envs + 1)
        transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
            step_keys[1:], state, actions.actions, params
        )
        return (transition.observations, transition.state, step_keys[0]), current

    _, recorded = jax.lax.scan(step, (observations, env_state, keys[0]), jnp.arange(rows // num_envs))
    return recorded.reshape(rows, recorded.shape[-1]).astype(jnp.int32)


def initialize_state[Model: nnx.Module](
    model: Model, buffers: Buffers, config: Schedule, key: Array
) -> ExplanationState[Model]:
    return ExplanationState(
        model,
        nnx.clone(model),
        make_adam(model, config.learning_rate, config.adam),
        buffers,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(jnp.inf, jnp.float32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(False),
        jnp.asarray(ExplanationPhase.TRAINING, jnp.int32),
        key,
    )


def template_state(context: RunContext, config: RunConfig) -> AnyState:
    """Host template with the run's static shapes, used only to validate and place restored arrays."""
    training = config.train
    features = context.features
    buffers = Buffers(
        jnp.zeros((training.train_size, features), jnp.int32),
        jnp.zeros((training.validation_size, features), jnp.int32),
        jnp.zeros((context.exact_rows, features), jnp.int32),
    )
    key = jax.random.key(0)
    if isinstance(config, CharacteristicRunConfig):
        network = characteristic_network(features, context.actions, config.characteristic)
        return initialize_state(CharacteristicModel(network, rngs=nnx.Rngs(0)), buffers, training, key)
    if isinstance(config, OutcomeRunConfig):
        networks = OutcomeNetworks(outcome_network(features, context.actions, config.outcome), rngs=nnx.Rngs(0))
        return initialize_state(networks, buffers, config.train, key)
    layout = ShapleyLayout(features, context.actions if context.explained.kind == "behaviour" else 1)
    model = ShapleyModel(shapley_network(layout, config.hidden_widths), layout, rngs=nnx.Rngs(0))
    return initialize_state(model, buffers, training, key)


def make_snapshot[Model: nnx.Module](state: ExplanationState[Model]) -> ExplanationSnapshot:
    """Extract numerical state only; NNX graph definitions stay outside callbacks."""
    from jax_shapley.explanation_io import ExplanationSnapshot

    return ExplanationSnapshot(
        _array_leaves(nnx.state(state.model)),
        _array_leaves(nnx.state(state.best)),
        _array_leaves(nnx.state(state.optimizer)),
        (state.buffers.train, state.buffers.validation, state.buffers.exact),
        (state.best_loss, state.best_epoch, state.stopped, jax.random.key_data(state.key)),
        state.epoch,
        state.phase,
        str(jax.random.key_impl(state.key)),
    )


def parse_args(args: list[str] | None = None) -> CLIConfig:
    return tyro.cli(CLIConfig, args=args)


def _bind(context: RunContext, config: RunConfig, sink: ExplanationSink) -> Callable[..., object]:
    """Bind static configuration, preserving each environment's state/parameter types."""
    env = context.env_config
    if isinstance(config, CharacteristicRunConfig):
        options = {"config": config, "max_unique": context.max_unique, "sink": sink}
        if isinstance(env, GridworldConfig):
            return partial(run_characteristic, environment_factory=Gridworld, **options)
        if isinstance(env, Mastermind222Config):
            return partial(run_characteristic, environment_factory=Mastermind222, **options)
        return partial(run_characteristic, environment_factory=env.make, **options)
    if isinstance(config, OutcomeRunConfig):
        # The transition model is a host-built constant shared by every seed.
        tabular = transition_model(env) if config.train.exact_loss else None
        outcome = {"config": config, "explained": context.explained, "tabular": tabular, "sink": sink}
        if isinstance(env, GridworldConfig):
            return partial(run_outcome, environment_factory=Gridworld, **outcome)
        if isinstance(env, Mastermind222Config):
            return partial(run_outcome, environment_factory=Mastermind222, **outcome)
        return partial(run_outcome, environment_factory=env.make, **outcome)
    options = {"config": config, "explained": context.explained, "max_unique": context.max_unique, "sink": sink}
    if isinstance(env, GridworldConfig):
        return partial(run_shapley, environment_factory=Gridworld, **options)
    if isinstance(env, Mastermind222Config):
        return partial(run_shapley, environment_factory=Mastermind222, **options)
    return partial(run_shapley, environment_factory=env.make, **options)


def _stack_optional[Tree](items: list[Tree | None]) -> Tree | None:
    present = [item for item in items if item is not None]
    if not present:
        return None
    if len(present) != len(items):
        raise ValueError("every seed needs the same frozen inputs")
    return stack_seeds(present)


def _flush(metrics: EpochMetrics, sink: ExplanationSink) -> None:
    jax.lax.cond(
        jnp.any(metrics.active),
        lambda: io_callback(sink.epochs, jax.ShapeDtypeStruct((), jnp.int32), _array_leaves(metrics), ordered=True),
        lambda: jnp.asarray(0, jnp.int32),
    )


def _emit_checkpoint[Model: nnx.Module](state: ExplanationState[Model], due: Array, sink: ExplanationSink) -> Array:
    """Commit the seeds marked `due`; snapshot leaves carry the seed axis."""
    return cast(
        Array,
        io_callback(sink.checkpoint, jax.ShapeDtypeStruct((), jnp.int32), make_snapshot(state), due, ordered=True),
    )  # Callback result has the declared scalar int32 specification.


def _copy_parameters(source: nnx.Module, destination: nnx.Module) -> None:
    nnx.update(destination, nnx.state(source, nnx.Param))


def _inactive_metrics() -> EpochMetrics:
    zero, nothing = jnp.asarray(0, jnp.int32), jnp.asarray(0.0, jnp.float32)
    return EpochMetrics(zero, zero, nothing, nothing, nothing, zero, jnp.asarray(False))


def _dimensions[State, Params](env: EpisodeStatisticsWrapper[State, Params]) -> tuple[int, int]:
    shape = env.observation_space(env.default_params).shape
    if len(shape) != 1:
        raise ValueError("explanations require flat observations")
    return int(shape[0]), env.num_actions


def _stage(config: RunConfig) -> str:
    if isinstance(config, CharacteristicRunConfig):
        return "characteristic"
    return "outcome characteristic" if isinstance(config, OutcomeRunConfig) else "Shapley model"


def _array_leaves(tree: object) -> tuple[Array, ...]:
    return tuple(cast(list[Array], jax.tree.leaves(tree)))  # Only numerical structs and NNX state cross this boundary.


def _run_options(seed: int, seeds: tuple[int, ...], chunk_epochs: int, validate_only: bool, compile_only: bool) -> None:
    _positive("chunk_epochs", chunk_epochs)
    if any(not 0 <= value <= 2**32 - 1 for value in (seed, *seeds)):
        raise ValueError("seeds must be in [0, 2**32 - 1]")
    if len(set(seeds)) != len(seeds):
        raise ValueError("sweep seeds must be distinct")
    if validate_only and compile_only:
        raise ValueError("validate_only and compile_only are mutually exclusive")


def _positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


if __name__ == "__main__":
    # Run the canonical module, so host I/O and this entry point share one set of config classes.
    from jax_shapley import train_explanations

    train_explanations.main()
