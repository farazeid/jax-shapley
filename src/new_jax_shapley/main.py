"""Run one experiment end to end: `python -m new_jax_shapley.main NAME [--track wandb|trackio] [overrides]`.

An experiment is the paper's linear pipeline for one explanation kind:

1. train the agent (`algorithms`);
2-3. roll out the frozen agent and train the characteristic models (`characteristic_models`);
4. train Shapley models explaining one characteristic seed (`shapley_models`).

Configs live in `new_jax_shapley/configs/NAME.py` as a module-level `CONFIG`; any field can be
overridden on the command line, e.g. `--algorithm.total-timesteps 5000` or `--shapley.1.train.epochs 9`.

`--compile-only` traces, lowers and compiles every stage's programs at the configured
shapes, then exits without running or tracking anything.

Each stage tracks itself: one run per seed named `NAME--STAGE--VARIANT[--seed-N]`, all in
one run group `NAME-TIMESTAMP`, and its whole final state pushed as an Orbax checkpoint
artifact. A stage given a `checkpoint` artifact (e.g. `--algorithm.checkpoint
GROUP--algorithm--dqn:latest`) loads it instead of training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from time import time_ns

import jax
import jax.numpy as jnp
import tyro
from flax import nnx

from new_jax_shapley import backends
from new_jax_shapley.algorithms import dqn
from new_jax_shapley.characteristic_models import behaviour as characteristic
from new_jax_shapley.characteristic_models import prediction as characteristic_prediction
from new_jax_shapley.common.supervised_learning import EpochMetrics, TrainState, seed_slice, stack_seeds
from new_jax_shapley.experiment import ExperimentConfig, load_configs
from new_jax_shapley.shapley_models import behaviour as shapley
from new_jax_shapley.shapley_models import prediction as shapley_prediction


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    dqn_state: dqn.RunState
    dqn_metrics: dqn.StepMetrics | None
    characteristic_state: TrainState[characteristic.CharacteristicModel]
    characteristic_metrics: EpochMetrics | None
    shapley: tuple[tuple[TrainState[shapley.ShapleyModel], EpochMetrics | None], ...]


def main(args: list[str] | None = None) -> str:
    """Run the named experiment, tracked; returns its run group (empty when only compiling)."""
    config = tyro.extras.overridable_config_cli(load_configs(), args=args)
    if config.compile_only:
        compile_stages(config)
        print(f"Compiled {config.name}.")
        return ""
    group = f"{config.name}-{time_ns()}"
    # One scratch folder for the whole experiment: later stages reuse earlier stages' files.
    with backends.session(config.track, config.entity):
        run(config, group)
    print(f"Finished {config.name}: {config.track} group {group} in project {config.project}.")
    return group


def run(config: ExperimentConfig, group: str | None = None) -> ExperimentResult:
    """The pipeline. With a `group` every stage is tracked under the experiment's settings;
    without one it runs untracked and in memory (tests)."""
    algorithm, characteristic_config, shapley_configs = config.algorithm, config.characteristic, config.shapley
    if group is not None:
        algorithm = replace(
            algorithm,
            name=config.name,
            track=config.track,
            project=config.project,
            entity=config.entity,
            group=group,
        )
        characteristic_config = replace(
            characteristic_config,
            name=config.name,
            track=config.track,
            project=config.project,
            entity=config.entity,
            group=group,
        )
        shapley_configs = tuple(
            replace(
                stage,
                name=config.name,
                track=config.track,
                project=config.project,
                entity=config.entity,
                group=group,
            )
            for stage in shapley_configs
        )

    # 1. Train the agents: one, or one per characteristic seed.
    dqn_state, dqn_metrics, agents = dqn.main(algorithm)

    # 2-3. Train characteristic models explaining them.
    if characteristic_config.kind == "behaviour":
        characteristic_state, characteristic_metrics = characteristic.main(characteristic_config, agents)
    else:
        characteristic_state, characteristic_metrics = characteristic_prediction.main(characteristic_config, agents)

    # 4. Train Shapley models explaining characteristic seed `explained_seed` (and its agent).
    index = characteristic_config.seeds.index(config.explained_seed)
    explained = seed_slice(characteristic_state, index)
    agent = seed_slice(agents, index if len(algorithm.seeds) > 1 else 0)
    shapley_main = shapley.main if characteristic_config.kind == "behaviour" else shapley_prediction.main
    shapley_results = tuple(
        shapley_main(stage, characteristic_config, agent, explained.model, explained.buffers)
        for stage in shapley_configs
    )
    return ExperimentResult(dqn_state, dqn_metrics, characteristic_state, characteristic_metrics, shapley_results)


def compile_stages(config: ExperimentConfig) -> None:
    """Trace, lower and compile every stage's programs at the configured shapes, with live-logging
    callbacks; nothing runs and nothing is tracked.

    Exact tables are sized by an upper bound on unique states (the exact buffer's rows), so
    the compiled shapes can exceed a real run's, which sizes them from its buffers.
    """
    key = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)
    algorithm, stage = config.algorithm, config.characteristic
    program = nnx.jit(partial(dqn.train, algorithm, lambda seed, metrics: None, lambda seed, metrics: None))
    program.trace(key, jax.ShapeDtypeStruct((), jnp.int32)).lower().compile()

    targets, head = (
        (characteristic.targets, characteristic.head)
        if stage.kind == "behaviour"
        else (characteristic_prediction.targets, characteristic_prediction.head)
    )
    seeds = len(stage.seeds)
    keys = jax.ShapeDtypeStruct((seeds,), key.dtype)
    agent = dqn.trained_agent(algorithm, dqn.initialize(algorithm, jax.random.key(0)))
    agents = stack_seeds([agent] * seeds)
    template = characteristic.initialize(stage, head)
    nnx.jit(partial(characteristic.collect, stage)).trace(keys, agents).lower().compile()
    program = nnx.jit(partial(characteristic.fit, stage, targets, head, stage.train.exact_size, lambda metrics: None))
    program.trace(keys, agents, template.buffers).lower().compile()

    explained = seed_slice(template, 0)
    for shapley_config in config.shapley:
        count = len(shapley_config.seeds)
        program = nnx.jit(
            partial(shapley.fit, shapley_config, stage, targets, head, stage.train.exact_size, lambda metrics: None)
        )
        program.trace(
            jax.ShapeDtypeStruct((count,), key.dtype),
            stack_seeds([agent] * count),
            stack_seeds([explained.model] * count),
            stack_seeds([explained.buffers] * count),
        ).lower().compile()


if __name__ == "__main__":
    main()
