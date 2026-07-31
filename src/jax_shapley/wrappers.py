"""JAX-native wrappers for environment rollout instrumentation."""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class EpisodeStatisticsState[EnvState]:
    """Wrapped environment state and per-environment episode accumulators."""

    env_state: EnvState
    episode_return: jax.Array
    episode_length: jax.Array
    episode_count: jax.Array


class EpisodeStatisticsWrapper[EnvState]:
    """Record fixed-shape episode statistics around an auto-resetting environment."""

    def __init__(self, env: Any):
        self._env = env

    @property
    def env(self) -> Any:
        """Return the unwrapped environment."""
        return self._env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any = None) -> tuple[jax.Array, EpisodeStatisticsState]:
        observation, env_state = self._env.reset(key, params)
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
        action: int | float | jax.Array,
        params: Any = None,
    ) -> tuple[
        jax.Array,
        EpisodeStatisticsState[EnvState],
        jax.Array,
        jax.Array,
        dict[str, Any],
    ]:
        observation, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        episode_return = state.episode_return + reward
        episode_length = state.episode_length + jnp.asarray(1, dtype=jnp.int32)
        completed = jnp.asarray(info["terminated"] | info["truncated"], dtype=jnp.bool_)

        next_state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.where(completed, 0.0, episode_return).astype(jnp.float32),
            episode_length=jnp.where(completed, 0, episode_length).astype(jnp.int32),
            episode_count=state.episode_count + completed.astype(jnp.int32),
        )
        episode_info = {
            "return": jnp.where(completed, episode_return, 0.0).astype(jnp.float32),
            "length": jnp.where(completed, episode_length, 0).astype(jnp.int32),
            "completed": completed,
        }
        return observation, next_state, reward, done, {**info, "episode": episode_info}
