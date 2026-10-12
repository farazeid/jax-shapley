"""The source's `train_models` epoch loop, shared by characteristic and Shapley models.

Every seed trains in one compiled program: state, contexts and frozen models carry a
leading seed axis. A model module supplies one `loss` function computing a sampled
train or validation batch's loss, or the exact loss, by its `partition` argument, and
optionally its own training `update` (default: one Adam step on the train loss). An
epoch draws `batches` training batches with replacement, then a
validation loss, an optional exact loss, and early stopping on validation. Finished
seeds no-op while others continue; at the end every seed restores its best-validation
model. The loop returns every epoch's metrics, (epochs, seeds) along the leading axes.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import partial
from math import isfinite, nan
from operator import itemgetter
from typing import Literal, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import trackio
import tyro
import wandb
from flax import nnx, struct
from jax import Array

from new_jax_shapley.common.networks import AdamConfig

type Partition = Literal["train", "validation", "exact"]
type Loss[Model, Context, Frozen, Config] = Callable[[Model, Frozen, Context, Config, int, Array, Partition], Array]
type Update[Model, Context, Frozen, Config] = Callable[
    [Model, nnx.Optimizer[Model], Frozen, Context, Config, int, Array], Array
]


class Schedule(Protocol):
    """The epoch loop settings `train` reads (`SupervisedConfig`, or a model's own training config)."""

    @property
    def epochs(self) -> int: ...

    @property
    def batches(self) -> int: ...

    @property
    def patience_epochs(self) -> int: ...

    @property
    def exact_loss(self) -> bool: ...

    @property
    def exact_condition(self) -> float | None: ...

    @property
    def track_batch(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class SupervisedConfig:
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
        if any(getattr(self, item.name) is tyro.MISSING for item in fields(self)):
            return  # A sweep config: validated once the command line supplies every field.
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("buffer_size", self.buffer_size),
            ("num_envs", self.num_envs),
            ("patience", self.epochs if self.patience is None else self.patience),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        exact = [] if self.exact_buffer_size is None else [("exact", self.exact_buffer_size)]
        for name, size in [("train", self.train_size), ("validation", self.validation_size), *exact]:
            if size < self.num_envs or size % self.num_envs:
                raise ValueError(f"{name} buffer size {size} must be a positive multiple of num_envs")
        if self.batches < 1:
            raise ValueError("the train buffer must hold at least one batch")
        if self.exact_condition is not None and (not self.exact_loss or not self.exact_condition > 0):
            raise ValueError("exact_condition must be positive and requires exact_loss")

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


@struct.dataclass
class ReplayBehaviour:
    """Actions and behaviour-policy log-probabilities of replay-drawn train and validation rows."""

    train_actions: Array
    train_log_probabilities: Array
    validation_actions: Array
    validation_log_probabilities: Array


@struct.dataclass
class StateBuffers:
    """int32 observations a frozen policy visited: train, validation, and the exact buffer
    exact values are computed over (`common.rollouts` fills them). Train and validation rows
    drawn from an agent's replay instead carry its `behaviour`."""

    train: Array
    validation: Array
    exact: Array
    behaviour: ReplayBehaviour | None = None


@struct.dataclass
class TrainState[Model]:
    model: Model
    best: Model
    optimizer: nnx.Optimizer[Model]
    buffers: StateBuffers
    epoch: Array
    best_loss: Array
    best_epoch: Array
    stopped: Array
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


def train[Model: nnx.Module, Context, Frozen, Config: Schedule](
    state: TrainState[Model],
    contexts: Context,
    frozen: Frozen,
    loss: Loss[Model, Context, Frozen, Config],
    config: Config,
    features: int,
    log: Callable[[EpochMetrics], None] | None,
    update: Update[Model, Context, Frozen, Config] | None = None,
) -> tuple[TrainState[Model], EpochMetrics]:
    """Every seed's epochs until all stop, then each seed's best-validation model.

    An epoch draws `batches` training batches with replacement, then a validation loss,
    an optional exact loss, and early stopping on validation. Finished seeds no-op while
    others continue. Returns every epoch's metrics, (epochs, seeds) along the leading axes.
    `log` receives each epoch's metrics for every seed during the run (a host callback);
    seeds that have stopped are marked inactive.
    """
    shape = (config.epochs, state.epoch.shape[0])
    integers, floats = jnp.zeros(shape, jnp.int32), jnp.zeros(shape, jnp.float32)
    empty = EpochMetrics(integers, integers, floats, floats, floats, integers, jnp.zeros(shape, jnp.bool_))

    def epoch(
        loss: Loss[Model, Context, Frozen, Config],
        update: Update[Model, Context, Frozen, Config] | None,
        config: Config,
        features: int,
        state: TrainState[Model],
        context: Context,
        frozen: Frozen,
    ) -> tuple[TrainState[Model], EpochMetrics]:
        """One seed's source epoch: shuffled-with-replacement batches, validation, exact loss, early stopping."""
        key, batch_key, validation_key = jax.random.split(state.key, 3)

        # The frozen model rides in the scan carry: NNX transforms only accept graph nodes
        # created at their own trace level as arguments.
        type BatchCarry = tuple[Model, nnx.Optimizer[Model], Frozen]

        def batch(
            update: Update[Model, Context, Frozen, Config],
            config: Config,
            features: int,
            context: Context,
            carry: BatchCarry,
            key: Array,
        ) -> tuple[BatchCarry, Array]:
            model, optimizer, frozen = carry
            return (model, optimizer, frozen), update(model, optimizer, frozen, context, config, features, key)

        step = partial(gradient_step, loss) if update is None else update
        _, losses = nnx.scan(
            partial(batch, step, config, features, context), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )((state.model, state.optimizer, frozen), jax.random.split(batch_key, config.batches))
        validation = loss(state.model, frozen, context, config, features, validation_key, "validation")
        exact = (
            loss(state.model, frozen, context, config, features, validation_key, "exact")
            if config.exact_loss
            else jnp.asarray(nan, jnp.float32)
        )
        number = state.epoch + 1
        improved = validation < state.best_loss
        nnx.cond(
            improved,
            lambda model, best: nnx.update(best, nnx.state(model, nnx.Param)),
            lambda model, best: None,
            state.model,
            state.best,
        )
        best_loss = jnp.where(improved, validation, state.best_loss)
        best_epoch = jnp.where(improved, number, state.best_epoch)
        stopped = number - best_epoch >= config.patience_epochs
        if config.exact_condition is not None:
            stopped = stopped | (exact < config.exact_condition)
        # Source logging step: the epoch, or (epoch + 1) * batches under track_batch.
        step = (number + 1) * config.batches if config.track_batch else number
        next_state = TrainState(
            state.model, state.best, state.optimizer, state.buffers, number, best_loss, best_epoch, stopped, key
        )
        metrics = EpochMetrics(number, step, jnp.mean(losses), validation, exact, best_epoch, jnp.asarray(True))
        return next_state, metrics

    def skip(state: TrainState[Model], context: Context, frozen: Frozen) -> tuple[TrainState[Model], EpochMetrics]:
        del context, frozen
        zero, nothing = jnp.asarray(0, jnp.int32), jnp.asarray(0.0, jnp.float32)
        return state, EpochMetrics(zero, zero, nothing, nothing, nothing, zero, jnp.asarray(False))

    def seed_epoch(
        loss: Loss[Model, Context, Frozen, Config],
        update: Update[Model, Context, Frozen, Config] | None,
        config: Config,
        features: int,
        state: TrainState[Model],
        context: Context,
        frozen: Frozen,
    ) -> tuple[TrainState[Model], EpochMetrics]:
        return nnx.cond(
            (state.epoch < config.epochs) & ~state.stopped,
            partial(epoch, loss, update, config, features),
            skip,
            state,
            context,
            frozen,
        )

    # Contexts and frozen models travel in the loop carry: NNX transforms only accept graph
    # nodes at their own trace level.
    type Carry = tuple[TrainState[Model], Context, Frozen, EpochMetrics, Array]

    def body(
        loss: Loss[Model, Context, Frozen, Config],
        update: Update[Model, Context, Frozen, Config] | None,
        config: Config,
        features: int,
        log: Callable[[EpochMetrics], None] | None,
        carry: Carry,
    ) -> Carry:
        state, contexts, frozen, metrics, index = carry
        every_seed = nnx.vmap(
            partial(seed_epoch, loss, update, config, features),
            in_axes=(0, 0, None if frozen is None else 0),
            out_axes=(0, 0),
        )
        state, epoch_metrics = every_seed(state, contexts, frozen)
        if log is not None:
            jax.debug.callback(log, epoch_metrics, ordered=True)
        rows = zip(jax.tree.leaves(metrics), jax.tree.leaves(epoch_metrics), strict=True)
        recorded = EpochMetrics(*[history.at[index].set(row) for history, row in rows])
        return state, contexts, frozen, recorded, index + 1

    def running(config: Config, carry: Carry) -> Array:
        return jnp.any((carry[0].epoch < config.epochs) & ~carry[0].stopped)

    initial: Carry = (state, contexts, frozen, empty, jnp.asarray(0, jnp.int32))
    looped = cast(
        Carry, nnx.while_loop(partial(running, config), partial(body, loss, update, config, features, log), initial)
    )
    trained, _, _, metrics, _ = looped

    def restore_best(state: TrainState[Model]) -> TrainState[Model]:
        nnx.update(state.model, nnx.state(state.best, nnx.Param))
        return state

    return nnx.vmap(restore_best, in_axes=0, out_axes=0)(trained), metrics


def gradient_step[Model: nnx.Module, Context, Frozen, Config](
    loss: Loss[Model, Context, Frozen, Config],
    model: Model,
    optimizer: nnx.Optimizer[Model],
    frozen: Frozen,
    context: Context,
    config: Config,
    features: int,
    key: Array,
) -> Array:
    """The default training update: one Adam step on a sampled train batch's loss; returns that loss."""
    # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
    value, gradients = cast(
        tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(loss)(model, frozen, context, config, features, key, "train"),
    )
    optimizer.update(model, gradients)
    return value


def log_epochs(runs: tuple[wandb.Run | trackio.Run, ...], model: str, exact_loss: bool, metrics: EpochMetrics) -> None:
    """The source's train_models scalars, `model/{model}_{epoch,val,exact}_loss`, into each seed's run
    (a host callback for `train`'s `log`; stopped seeds log nothing)."""
    active, steps = np.asarray(metrics.active), np.asarray(metrics.step)
    train_loss, validation = np.asarray(metrics.train_loss), np.asarray(metrics.validation_loss)
    exact = np.asarray(metrics.exact_loss)
    for index, run in enumerate(runs):
        if not active[index]:
            continue
        row = {
            f"model/{model}_epoch_loss": float(train_loss[index]),
            f"model/{model}_val_loss": float(validation[index]),
        }
        if exact_loss:
            row[f"model/{model}_exact_loss"] = float(exact[index])
        run.log(row, step=int(steps[index]))


def stack_seeds[Tree](items: list[Tree]) -> Tree:
    """Stack equally shaped pytrees (NNX modules included) along a new leading seed axis."""
    graph, first = nnx.split(items[0])
    states = [first, *(nnx.split(item)[1] for item in items[1:])]
    return nnx.merge(graph, jax.tree.map(lambda *leaves: jnp.stack(leaves), *states))


def seed_slice[Tree](tree: Tree, seed: int) -> Tree:
    """One seed of a seed-stacked pytree."""
    graph, state = nnx.split(tree)
    return nnx.merge(graph, jax.tree.map(itemgetter(seed), state))
