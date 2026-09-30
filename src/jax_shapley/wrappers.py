"""JAX-native wrappers for environment rollout instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from jax_shapley.envs.contracts import Counter, EpisodeInfo, EpisodeStepInfo, Flag, Observation, Reward, StepInfo

__all__ = ["EnvironmentProtocol", "EpisodeInfo", "EpisodeStatisticsState", "EpisodeStatisticsWrapper"]


class EnvironmentProtocol[EnvState, EnvParams](Protocol):
    """Minimal typed interface expected by the episode-statistics wrapper."""

    @property
    def default_params(self) -> EnvParams: ...

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[Observation, EnvState]: ...

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: float | jax.Array,
        params: EnvParams,
    ) -> tuple[Observation, EnvState, Reward, Flag, StepInfo]: ...

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[Observation, EnvState]: ...

    def step_env(
        self, key: jax.Array, state: EnvState, action: float | jax.Array, params: EnvParams
    ) -> tuple[Observation, EnvState, Reward, Flag, StepInfo]: ...

    @property
    def name(self) -> str: ...

    def observation_space(self, params: EnvParams) -> spaces.Box: ...

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete: ...

    @property
    def num_actions(self) -> int: ...


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Typed episode metadata produced by the wrapper around each step."""

    return_value: Reward
    length: Counter
    completed: Flag


@struct.dataclass
class EpisodeStatisticsState[EnvState]:
    """Wrapped environment state and per-environment episode accumulators."""

    env_state: EnvState
    episode_return: Reward
    episode_length: Counter
    episode_count: Counter


class EpisodeStatisticsWrapper[EnvState, EnvParams]:
    """Record fixed-shape episode statistics around an auto-resetting environment."""

    def __init__(self, env: EnvironmentProtocol[EnvState, EnvParams]) -> None:
        self._env = env

    @property
    def env(self) -> EnvironmentProtocol[EnvState, EnvParams]:
        """Return the unwrapped environment."""

        return self._env

    @property
    def default_params(self) -> EnvParams:
        return self._env.default_params

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return self._env.observation_space(params)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return self._env.action_space(params)

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    @property
    def name(self) -> str:
        return self._env.name

    @staticmethod
    def _defaulted_params(
        env: EnvironmentProtocol[EnvState, EnvParams],
        params: EnvParams | None,
    ) -> EnvParams:
        return env.default_params if params is None else params

    @staticmethod
    def _as_episode_info(
        completed: Flag,
        episode_return: Reward,
        episode_length: Counter,
    ) -> EpisodeInfo:
        return {
            "return": episode_return,
            "length": episode_length,
            "completed": completed,
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[Observation, EpisodeStatisticsState[EnvState]]:
        resolved_params = self._defaulted_params(self._env, params)
        observation, env_state = self._env.reset(key, resolved_params)
        state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            episode_count=jnp.asarray(0, dtype=jnp.int32),
        )
        return observation, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: jax.Array,
        state: EpisodeStatisticsState[EnvState],
        action: float | jax.Array,
        params: EnvParams | None = None,
    ) -> tuple[
        Observation,
        EpisodeStatisticsState[EnvState],
        Reward,
        Flag,
        EpisodeStepInfo,
    ]:
        resolved_params = self._defaulted_params(self._env, params)
        observation, env_state, reward, done, info = self._env.step(key, state.env_state, action, resolved_params)
        episode_return = state.episode_return + reward
        episode_length = state.episode_length + jnp.asarray(1, dtype=jnp.int32)
        completed = jnp.asarray(info["terminated"] | info["truncated"], dtype=jnp.bool_)

        next_state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.where(completed, 0.0, episode_return).astype(jnp.float32),
            episode_length=jnp.where(completed, 0, episode_length).astype(jnp.int32),
            episode_count=state.episode_count + completed.astype(jnp.int32),
        )

        summary = EpisodeSummary(
            return_value=jnp.where(completed, episode_return, 0.0).astype(jnp.float32),
            length=jnp.where(completed, episode_length, 0).astype(jnp.int32),
            completed=completed,
        )
        episode_info = self._as_episode_info(
            completed=summary.completed,
            episode_return=summary.return_value,
            episode_length=summary.length,
        )
        wrapped_info: EpisodeStepInfo = {
            "discount": info["discount"],
            "terminated": info["terminated"],
            "truncated": info["truncated"],
            "episode": episode_info,
        }
        return (
            observation,
            next_state,
            reward,
            done,
            wrapped_info,
        )
