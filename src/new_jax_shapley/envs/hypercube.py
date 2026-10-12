"""JAX port of FastSVERL's obstacle-free Hypercube environment."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

from new_jax_shapley.envs.base import TypedEnvironment
from new_jax_shapley.envs.contracts import Counter, Flag, Observation, Reward, StepInfo

if TYPE_CHECKING:
    import numpy as np

    from new_jax_shapley.envs.render import RenderConfig


@struct.dataclass
class HypercubeState(environment.EnvState):
    """Hypercube state carried through JAX transformations."""

    position: Observation
    terminated: Flag
    time: Counter


@struct.dataclass
class HypercubeParams(environment.EnvParams):
    """Hypercube parameters; the paper experiments have no time limit."""

    max_steps_in_episode: int = 2**31 - 1


@dataclass(frozen=True, slots=True)
class Hypercube(TypedEnvironment[HypercubeState, HypercubeParams]):
    """An obstacle-free ``n``-sided, ``d``-dimensional gridworld."""

    side_length: int
    dimension: int

    def __post_init__(self) -> None:
        if self.side_length not in range(2, 7):
            raise ValueError("side_length must be one of the paper values 2, 3, 4, 5, or 6")
        if self.dimension not in range(2, 6):
            raise ValueError("dimension must be one of the paper values 2, 3, 4, or 5")

    @property
    def default_params(self) -> HypercubeParams:
        return HypercubeParams()

    def reset_env(
        self,
        key: jax.Array,
        params: HypercubeParams,
    ) -> tuple[Observation, HypercubeState]:
        del key, params
        state = HypercubeState(
            position=jnp.zeros(self.dimension, dtype=jnp.int32),
            terminated=jnp.asarray(False),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def step_env(
        self,
        key: jax.Array,
        state: HypercubeState,
        action: float | jax.Array,
        params: HypercubeParams,
    ) -> tuple[Observation, HypercubeState, Reward, Flag, StepInfo]:
        del key, params
        action = jnp.asarray(action, dtype=jnp.int32)
        axis = action // 2
        direction = jnp.where(action % 2 == 0, 1, -1).astype(jnp.int32)
        candidate = state.position.at[axis].add(direction)
        valid = jnp.all((candidate >= 0) & (candidate < self.side_length))
        position = jnp.where(valid, candidate, state.position)
        terminated = jnp.all(position == jnp.full(self.dimension, self.side_length - 1, dtype=jnp.int32))
        reward = jnp.where(terminated, float(self.side_length * self.dimension - 1), -1.0).astype(jnp.float32)
        next_state = HypercubeState(
            position=position,
            terminated=terminated,
            time=state.time + jnp.asarray(1, dtype=jnp.int32),
        )
        info: StepInfo = {
            "discount": jnp.where(terminated, 0.0, 1.0).astype(jnp.float32),
            "terminated": terminated,
            "truncated": jnp.asarray(False),
        }
        return self.get_obs(next_state), next_state, reward, terminated, info

    def render(
        self,
        state: HypercubeState,
        params: HypercubeParams | None = None,
        render_config: "RenderConfig | None" = None,
    ) -> "np.ndarray":
        """Render a 2D view and coordinate tracks as a host-side RGB array."""
        del params
        from new_jax_shapley.envs.render.hypercube import render_hypercube

        return render_hypercube(
            state,
            side_length=self.side_length,
            dimension=self.dimension,
            start=jnp.zeros(self.dimension, dtype=jnp.int32),
            goal=jnp.full(self.dimension, self.side_length - 1, dtype=jnp.int32),
            config=render_config,
        )

    @property
    def name(self) -> str:
        return "Hypercube-v0"

    @property
    def num_actions(self) -> int:
        return 2 * self.dimension

    def action_space(self, params: HypercubeParams | None = None) -> spaces.Discrete:
        del params
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: HypercubeParams) -> spaces.Box:
        del params
        return spaces.Box(
            low=0,
            high=self.side_length - 1,
            shape=(self.dimension,),
            dtype=jnp.int32,
        )

    def state_space(self, params: HypercubeParams) -> spaces.Dict:
        return spaces.Dict({
            "position": self.observation_space(params),
            "terminated": spaces.Discrete(2),
            "time": spaces.Discrete(params.max_steps_in_episode),
        })

    @overload
    def get_obs(self, state: HypercubeState) -> Observation: ...

    @overload
    def get_obs(self, state: HypercubeState, params: HypercubeParams) -> Observation: ...

    @overload
    def get_obs(
        self,
        state: HypercubeState,
        key: jax.Array,
        params: HypercubeParams,
    ) -> Observation: ...

    def get_obs(
        self,
        state: HypercubeState,
        *args: HypercubeParams | jax.Array,
        params: HypercubeParams | None = None,
        key: jax.Array | None = None,
    ) -> Observation:
        del args, params, key
        return state.position

    def is_terminal(self, state: HypercubeState, params: HypercubeParams) -> Flag:
        del params
        return state.terminated
