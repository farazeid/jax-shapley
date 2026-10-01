"""Shell-driven DQN: compile a complete numerical run before creating host writers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from time import time_ns
from typing import Annotated, Protocol, cast

import jax
import jax.numpy as jnp
import tyro
from flax import nnx
from jax import Array

from jax_shapley.algorithms.dqn import RunState, WholeRunState, continue_run, initialize_run, whole_run
from jax_shapley.artifacts import describe, initialize_metrics
from jax_shapley.checkpoints import CheckpointMetadata, CheckpointStore, restore_run
from jax_shapley.config import (
    DQNRunConfig,
    EnvironmentConfig,
    GridworldConfig,
    HypercubeConfig,
    Mastermind222Config,
    ResumeConfig,
)
from jax_shapley.execution import CallbackToken, EventSink, Snapshot
from jax_shapley.logging import ArtifactSink, verify_log
from jax_shapley.settings import load_settings
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsWrapper

type CLIConfig = (
    Annotated[
        DQNRunConfig[GridworldConfig], tyro.conf.subcommand(name="gridworld", default=DQNRunConfig(GridworldConfig()))
    ]
    | Annotated[
        DQNRunConfig[Mastermind222Config],
        tyro.conf.subcommand(name="mastermind222", default=DQNRunConfig(Mastermind222Config())),
    ]
    | Annotated[
        DQNRunConfig[HypercubeConfig], tyro.conf.subcommand(name="hypercube", default=DQNRunConfig(HypercubeConfig()))
    ]
    | Annotated[ResumeConfig, tyro.conf.subcommand(name="resume")]
)


@dataclass(frozen=True, slots=True)
class CompiledProgram[State]:
    fresh: FreshRunner[State] | None
    resume: ResumeRunner[State] | None
    router: CallbackRouter


@dataclass(frozen=True, slots=True)
class RunSummary:
    folder: str
    vector_steps: int
    environment_steps: int
    updates: int
    target_updates: int
    evaluation_episodes: int
    training_seconds: float
    backend: str


class FreshRunner[State](Protocol):
    def __call__(self, key: Array) -> WholeRunState[State]: ...


class ResumeRunner[State](Protocol):
    def __call__(self, state: RunState[State], phase: Array, evaluation_episodes: Array) -> WholeRunState[State]: ...


class CallbackRouter:
    """Bind the writer only after compilation; preflight cannot perform host I/O."""

    def __init__(self) -> None:
        self.sink: EventSink | None = None

    def training_complete(self) -> CallbackToken:
        return self._sink().training_complete()

    def training(self, metrics: tuple[Array, ...], count: Array) -> CallbackToken:
        return self._sink().training(metrics, count)

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> CallbackToken:
        return self._sink().evaluation(returns, completed, first_index)

    def checkpoint(self, snapshot: Snapshot) -> CallbackToken:
        return self._sink().checkpoint(snapshot)

    def _sink(self) -> EventSink:
        if self.sink is None:
            raise RuntimeError("callback invoked before writer binding")
        return self.sink


def main(args: list[str] | None = None) -> RunSummary | None:
    options = parse_args(args)
    resume: ResumeConfig | None = options if isinstance(options, ResumeConfig) else None
    config = load_settings(resume.run / "config.json") if resume is not None else options
    assert isinstance(config, DQNRunConfig)
    if config.validate_only:
        print(describe(config), end="")
        return None
    if isinstance(config.env, GridworldConfig):
        return run_experiment(config.env.make(), config, resume=resume)
    if isinstance(config.env, Mastermind222Config):
        return run_experiment(config.env.make(), config, resume=resume)
    return run_experiment(config.env.make(), config, resume=resume)


def run_experiment[State, Params, Config: EnvironmentConfig](
    environment: EnvironmentProtocol[State, Params],
    config: DQNRunConfig[Config],
    *,
    resume: ResumeConfig | None = None,
) -> RunSummary | None:
    env = EpisodeStatisticsWrapper(environment)
    metadata: CheckpointMetadata | None = None
    checkpoint_sequence: int | None = None
    restored: RunState[State] | None = None
    if resume is not None:
        store = CheckpointStore(resume.run, read_only=True)
        try:
            template = initialize_run(
                env, env.default_params, config.network, config.train, jax.random.key(config.seed)
            )
            restored, metadata = restore_run(store, template, config)
            checkpoint_sequence = store.sequence
            verify_log(resume.run / "metrics.csv", metadata)
        finally:
            store.close()
        if resume.validate_only:
            print(describe(config), end="")
            return None
    program = compile_program(env, env.default_params, config, resume_state=restored)
    if config.compile_only or (resume is not None and resume.compile_only):
        print(f"Compiled {env.name} DQN on {jax.default_backend()}; no run artifacts created.")
        return None
    folder = resume.run.resolve() if resume is not None else config.output / f"{env.name}-{config.seed}-{time_ns()}"
    if resume is None:
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "config.json").write_text(describe(config))
        initialize_metrics(folder / "metrics.csv")
    sink = ArtifactSink(folder, config, metadata=metadata, checkpoint_sequence=checkpoint_sequence)
    program.router.sink = sink
    try:
        if program.fresh is not None:
            result = program.fresh(jax.random.key(config.seed))
        else:
            assert program.resume is not None and restored is not None and metadata is not None
            result = program.resume(
                restored, jnp.asarray(metadata.phase, jnp.int32), jnp.asarray(metadata.evaluation_episodes, jnp.int32)
            )
        jax.block_until_ready(result.run.vector_step)
        jax.effects_barrier()
        if sink.store is not None:
            sink.store.wait()
        summary = RunSummary(
            str(folder),
            int(result.run.vector_step),
            int(result.run.vector_step) * config.train.num_envs,
            int(result.run.updates),
            int(result.run.target_updates),
            int(result.evaluation_episodes),
            sink.training_seconds,
            jax.default_backend(),
        )
        _write_summary(folder, summary)
    finally:
        sink.close()
        program.router.sink = None
    print(
        f"Trained {env.name}: {summary.environment_steps} transitions, {summary.updates} updates. Artifacts: {folder}"
    )
    return summary


def parse_args(args: list[str] | None = None) -> CLIConfig:
    return tyro.cli(CLIConfig, args=args)


def compile_program[State, Params, Config: EnvironmentConfig](
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    config: DQNRunConfig[Config],
    *,
    resume_state: RunState[State] | None = None,
) -> CompiledProgram[State]:
    router = CallbackRouter()
    key_spec = jax.ShapeDtypeStruct((), jax.random.key(config.seed).dtype)
    if resume_state is not None:
        # Retain the restored optimizer's static Optax functions in the compiled graph.
        abstract = nnx.eval_shape(lambda state: state, resume_state)
        runner = nnx.jit(
            partial(
                continue_run,
                env=env,
                params=params,
                config=config.train,
                chunk_steps=config.chunk_steps,
                checkpoint_mode=config.checkpoint_mode,
                sink=router,
            )
        )
        scalar = jax.ShapeDtypeStruct((), jnp.int32)
        compiled = cast(ResumeRunner[State], runner.trace(abstract, scalar, scalar).lower().compile())
        return CompiledProgram(None, compiled, router)
    runner = nnx.jit(
        partial(
            whole_run,
            env=env,
            params=params,
            network=config.network,
            config=config.train,
            chunk_steps=config.chunk_steps,
            checkpoint_mode=config.checkpoint_mode,
            sink=router,
        )
    )
    fresh = cast(FreshRunner[State], runner.trace(key_spec).lower().compile())
    return CompiledProgram(fresh, None, router)


def checkpoint_due(vector_step: int, num_envs: int, total_timesteps: int) -> bool:
    return (vector_step * num_envs) % max(1, total_timesteps // 10) < num_envs


def _write_summary(folder: Path, summary: RunSummary) -> None:
    staged = folder / "summary.json.tmp"
    staged.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    staged.replace(folder / "summary.json")


if __name__ == "__main__":
    main()
