"""Shell-driven DQN CLI: resolve, compile abstract state, then run host orchestration."""

import json
from dataclasses import asdict, dataclass
from functools import partial
from time import perf_counter, time_ns
from typing import Annotated, Protocol, cast

import jax
import jax.numpy as jnp
import tyro
from flax import nnx
from jax import Array

from jax_shapley.algorithms.dqn import (
    EvaluationResult,
    RunState,
    StepMetrics,
    evaluate_dqn,
    initialize_run,
    run_chunk,
)
from jax_shapley.artifacts import (
    MetricRow,
    describe,
    host_array,
    initialize_metrics,
    record_metrics,
    save_checkpoint,
    write_metric_rows,
)
from jax_shapley.config import (
    DQNRunConfig,
    EnvironmentConfig,
    GridworldConfig,
    HypercubeConfig,
    Mastermind222Config,
)
from jax_shapley.networks import QNetwork
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsWrapper

# The selected subcommand owns the whole run schema, avoiding argument-order
# surprises from a nested environment subcommand followed by root options.
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
)


class Initializer[State](Protocol):
    def __call__(self, key: Array) -> RunState[State]: ...


class ChunkRunner[State](Protocol):
    def __call__(self, state: RunState[State], active_steps: Array) -> tuple[RunState[State], StepMetrics]: ...


class Evaluator(Protocol):
    def __call__(self, model: QNetwork, key: Array) -> EvaluationResult: ...


@dataclass(frozen=True, slots=True)
class CompiledProgram[State]:
    initialize: Initializer[State]
    chunk: ChunkRunner[State]
    evaluate: Evaluator | None


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


def main(args: list[str] | None = None) -> RunSummary | None:
    config = parse_args(args)
    if config.validate_only:
        print(describe(config), end="")
        return None
    # Narrow the closed union before coupling each environment to its own params.
    if isinstance(config.env, GridworldConfig):
        return run_experiment(config.env.make(), config)
    if isinstance(config.env, Mastermind222Config):
        return run_experiment(config.env.make(), config)
    return run_experiment(config.env.make(), config)


def run_experiment[State, Params, Config: EnvironmentConfig](
    environment: EnvironmentProtocol[State, Params], config: DQNRunConfig[Config]
) -> RunSummary | None:
    env = EpisodeStatisticsWrapper(environment)
    program = compile_program(env, env.default_params, config)
    if config.compile_only:
        print(f"Compiled {env.name} DQN on {jax.default_backend()}; no run artifacts created.")
        return None
    state = program.initialize(jax.random.key(config.seed))
    folder = config.output / f"{env.name}-{config.seed}-{time_ns()}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(describe(config))
    metrics_path = folder / "metrics.csv"
    initialize_metrics(metrics_path)
    started = perf_counter()
    current = 0
    while current < config.train.vector_steps:
        active = _active_steps(current, config)
        state, metrics = program.chunk(state, jnp.asarray(active, jnp.int32))
        record_metrics(metrics_path, metrics, config.train.num_envs)
        current += active
        if checkpoint_due(current, config.train.num_envs, config.train.total_timesteps):
            save_checkpoint(folder, state, config)
    # Keep a final artifact even when floor division misses an upstream save event.
    save_checkpoint(folder, state, config)
    jax.block_until_ready(state.replay.size)
    training_seconds = perf_counter() - started
    evaluation_count = 0
    if program.evaluate is not None:
        evaluation = program.evaluate(state.model, state.key)
        evaluation_count = int(evaluation.count)
        returns = host_array(evaluation.returns)
        rows = [
            MetricRow("evaluation", index, episode_return=float(returns[index])) for index in range(evaluation_count)
        ]
        write_metric_rows(metrics_path, rows)
    summary = RunSummary(
        str(folder),
        current,
        current * config.train.num_envs,
        int(state.updates),
        int(state.target_updates),
        evaluation_count,
        training_seconds,
        jax.default_backend(),
    )
    (folder / "summary.json").write_text(json.dumps(asdict(summary), indent=2) + "\n")
    print(
        f"Trained {env.name}: {summary.environment_steps} transitions, {summary.updates} updates. Artifacts: {folder}"
    )
    return summary


def parse_args(args: list[str] | None = None) -> CLIConfig:
    return tyro.cli(CLIConfig, args=args)


def compile_program[State, Params, Config: EnvironmentConfig](
    env: EpisodeStatisticsWrapper[State, Params], params: Params, config: DQNRunConfig[Config]
) -> CompiledProgram[State]:
    key_spec = jax.ShapeDtypeStruct((), jax.random.key(config.seed).dtype)
    initialize = nnx.jit(partial(initialize_run, env, params, config.network, config.train))
    # NNX eval_shape retains graph semantics while leaving replay/model arrays abstract.
    abstract = nnx.eval_shape(initialize, key_spec)
    initialized = cast(Initializer[State], initialize.trace(key_spec).lower().compile())
    chunk = nnx.jit(partial(run_chunk, env=env, params=params, config=config.train, chunk_steps=config.chunk_steps))
    compiled = cast(ChunkRunner[State], chunk.trace(abstract, jax.ShapeDtypeStruct((), jnp.int32)).lower().compile())
    evaluate: Evaluator | None = None
    if config.train.eval_episodes:
        evaluation = nnx.jit(
            partial(
                evaluate_dqn,
                env=env,
                params=params,
                num_envs=config.train.num_envs,
                episodes=config.train.eval_episodes,
            )
        )
        evaluate = cast(Evaluator, evaluation.trace(abstract.model, key_spec).lower().compile())
    return CompiledProgram(initialized, compiled, evaluate)


def checkpoint_due(vector_step: int, num_envs: int, total_timesteps: int) -> bool:
    # Upstream divides by total_timesteps // 10. Guard only its host save cadence
    # for tiny smoke runs (<10 transitions), leaving learning semantics unchanged.
    interval = max(1, total_timesteps // 10)
    return (vector_step * num_envs) % interval < num_envs


def _active_steps[Config: EnvironmentConfig](current: int, config: DQNRunConfig[Config]) -> int:
    active = min(config.chunk_steps, config.train.vector_steps - current)
    if config.checkpoint_mode != "none":
        for offset in range(1, active + 1):
            if checkpoint_due(current + offset, config.train.num_envs, config.train.total_timesteps):
                return offset
    return active


if __name__ == "__main__":
    main()
