"""JAX-native wrappers for environment rollout instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Protocol, TypedDict

import jax
import jax.numpy as jnp
from flax import struct


class EnvironmentProtocol[EnvState, EnvParams](Protocol):
    """Minimal typed interface expected by the episode-statistics wrapper."""

    @property
    def default_params(self) -> EnvParams: ...

    def reset(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, EnvState]: ...

    def step(
        self,
        key: jax.Array,
        state: EnvState,
        action: float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[str, jax.Array]]: ...

    def observation_space(self, params: EnvParams) -> object: ...

    def action_space(self, params: EnvParams | None = None) -> object: ...

    @property
    def num_actions(self) -> int: ...


EpisodeInfo = TypedDict("EpisodeInfo", {"return": jax.Array, "length": jax.Array, "completed": jax.Array})


@dataclass(frozen=True)
class EpisodeSummary:
    """Typed episode metadata produced by the wrapper around each step."""

    return_value: jax.Array
    length: jax.Array
    completed: jax.Array


@struct.dataclass
class EpisodeStatisticsState[EnvState]:
    """Wrapped environment state and per-environment episode accumulators."""

    env_state: EnvState
    episode_return: jax.Array
    episode_length: jax.Array
    episode_count: jax.Array


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

    def observation_space(self, params: EnvParams) -> object:
        return self._env.observation_space(params)

    def action_space(self, params: EnvParams | None = None) -> object:
        return self._env.action_space(params)

    @property
    def num_actions(self) -> int:
        return self._env.num_actions

    def __getattr__(self, name: str) -> object:
        return getattr(self._env, name)

    @staticmethod
    def _defaulted_params(
        env: EnvironmentProtocol[EnvState, EnvParams],
        params: EnvParams | None,
    ) -> EnvParams:
        return env.default_params if params is None else params

    @staticmethod
    def _as_episode_info(
        completed: jax.Array,
        episode_return: jax.Array,
        episode_length: jax.Array,
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
    ) -> tuple[jax.Array, EpisodeStatisticsState[EnvState]]:
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
        jax.Array,
        EpisodeStatisticsState[EnvState],
        jax.Array,
        jax.Array,
        dict[str, jax.Array | EpisodeInfo],
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
        wrapped_info: dict[str, jax.Array | EpisodeInfo] = dict(info)
        wrapped_info["episode"] = episode_info
        return (
            observation,
            next_state,
            reward,
            done,
            wrapped_info,
        )
