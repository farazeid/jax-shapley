"""Immutable DQN contracts; shell recipes supply experiment-specific values."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from jax_shapley.envs import Gridworld, Hypercube, Mastermind222
from jax_shapley.networks import MLPConfig, dqn_config
from jax_shapley.wrappers import EnvironmentProtocol


def _positive(name: str, value: int) -> None:
    if value < 1 or value > 2**31 - 1:
        raise ValueError(f"{name} must be a positive int32 value")


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _probability(name: str, value: float) -> None:
    _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            _probability(name, value)
            if value == 1:
                raise ValueError(f"{name} must be less than 1")
        _finite("epsilon", self.epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True, slots=True)
class DQNTrainConfig:
    """Gridworld baseline; timestep/frequency units count all environment transitions."""

    num_envs: int = 1
    total_timesteps: int = 50_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 50_000
    gamma: float = 1.0
    tau: float = 0.01
    target_network_frequency: int = 1
    batch_size: int = 64
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10
    eval_episodes: int = 0
    adam: AdamConfig = AdamConfig()

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("total_timesteps", self.total_timesteps),
            ("buffer_size", self.buffer_size),
            ("batch_size", self.batch_size),
            ("train_frequency", self.train_frequency),
            ("target_network_frequency", self.target_network_frequency),
        ):
            _positive(name, value)
        if self.vector_steps < 1:
            raise ValueError("total_timesteps must include at least one vector step")
        for name, value in (("learning_starts", self.learning_starts), ("eval_episodes", self.eval_episodes)):
            if not 0 <= value <= 2**31 - 1:
                raise ValueError(f"{name} must be a nonnegative int32 value")
        for name, value in (("gamma", self.gamma), ("tau", self.tau), ("start_e", self.start_e), ("end_e", self.end_e)):
            _probability(name, value)
        for name, value in (("learning_rate", self.learning_rate), ("exploration_fraction", self.exploration_fraction)):
            _finite(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def vector_steps(self) -> int:
        return self.total_timesteps // self.num_envs


@dataclass(frozen=True, slots=True)
class GridworldConfig:
    def make(self) -> Gridworld:
        return Gridworld()


@dataclass(frozen=True, slots=True)
class Mastermind222Config:
    def make(self) -> Mastermind222:
        return Mastermind222()


@dataclass(frozen=True, slots=True)
class HypercubeConfig:
    side_length: int = 2
    dimension: int = 2

    def __post_init__(self) -> None:
        if self.side_length not in range(2, 7) or self.dimension not in range(2, 6):
            raise ValueError("Hypercube requires side_length 2..6 and dimension 2..5")

    def make(self) -> Hypercube:
        return Hypercube(self.side_length, self.dimension)


type EnvironmentConfig = GridworldConfig | Mastermind222Config | HypercubeConfig


@dataclass(frozen=True, slots=True)
class DQNRunConfig[EnvConfig: EnvironmentConfig]:
    env: EnvConfig
    train: DQNTrainConfig = DQNTrainConfig()
    hidden_widths: tuple[int, ...] = (64, 64)
    seed: int = 1
    chunk_steps: int = 64
    output: Path = Path("runs")
    save_model: bool = True
    save_replay: bool = True
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _positive("chunk_steps", self.chunk_steps)
        for width in self.hidden_widths:
            _positive("hidden_width", width)
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        if self.validate_only and self.compile_only:
            raise ValueError("validate_only and compile_only are mutually exclusive")

    @property
    def network(self) -> MLPConfig:
        return network_config(self.env, self.hidden_widths)


def network_config(config: EnvironmentConfig, widths: tuple[int, ...]) -> MLPConfig:
    if isinstance(config, GridworldConfig):
        return _network_for_env(config.make(), widths)
    if isinstance(config, Mastermind222Config):
        return _network_for_env(config.make(), widths)
    return _network_for_env(config.make(), widths)


def _network_for_env[State, Params](env: EnvironmentProtocol[State, Params], widths: tuple[int, ...]) -> MLPConfig:
    shape = env.observation_space(env.default_params).shape
    if len(shape) != 1:
        raise ValueError("DQN requires flat observations")
    return dqn_config(shape[0], env.num_actions, hidden_widths=widths)
