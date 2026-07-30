"""JAX port of FastSVERL's obstacle-free Hypercube environment."""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class HypercubeState(environment.EnvState):
    """Hypercube state carried through JAX transformations."""

    position: jax.Array
    terminated: jax.Array
    time: jax.Array


@struct.dataclass
class HypercubeParams(environment.EnvParams):
    """Hypercube parameters; the paper experiments have no time limit."""

    max_steps_in_episode: int = 2**31 - 1


class Hypercube(environment.Environment[HypercubeState, HypercubeParams]):
    """An obstacle-free ``n``-sided, ``d``-dimensional gridworld."""

    def __init__(self, side_length: int, dimension: int):
        if side_length not in range(2, 7):
            raise ValueError("side_length must be one of the paper values 2, 3, 4, 5, or 6")
        if dimension not in range(2, 6):
            raise ValueError("dimension must be one of the paper values 2, 3, 4, or 5")
        self.side_length = side_length
        self.dimension = dimension
        self._start = jnp.zeros(dimension, dtype=jnp.int32)
        self._goal = jnp.full(dimension, side_length - 1, dtype=jnp.int32)

    @property
    def default_params(self) -> HypercubeParams:
        return HypercubeParams()

    def reset_env(
        self, key: jax.Array, params: HypercubeParams
    ) -> tuple[jax.Array, HypercubeState]:
        del key, params
        state = HypercubeState(
            position=self._start,
            terminated=jnp.asarray(False),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def step_env(
        self,
        key: jax.Array,
        state: HypercubeState,
        action: int | float | jax.Array,
        params: HypercubeParams,
    ) -> tuple[jax.Array, HypercubeState, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, params
        action = jnp.asarray(action, dtype=jnp.int32)
        axis = action // 2
        direction = jnp.where(action % 2 == 0, 1, -1).astype(jnp.int32)
        candidate = state.position.at[axis].add(direction)
        valid = jnp.all((candidate >= 0) & (candidate < self.side_length))
        position = jnp.where(valid, candidate, state.position)
        terminated = jnp.all(position == self._goal)
        reward = jnp.where(terminated, float(self.side_length * self.dimension - 1), -1.0).astype(
            jnp.float32
        )
        next_state = HypercubeState(
            position=position,
            terminated=terminated,
            time=state.time + jnp.asarray(1, dtype=jnp.int32),
        )
        info = {
            "discount": jnp.where(terminated, 0.0, 1.0).astype(jnp.float32),
            "terminated": terminated,
            "truncated": jnp.asarray(False),
        }
        return self.get_obs(next_state), next_state, reward, terminated, info

    def get_obs(
        self,
        state: HypercubeState,
        params: HypercubeParams | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del params, key
        return state.position

    def is_terminal(self, state: HypercubeState, params: HypercubeParams) -> jax.Array:
        del params
        return state.terminated

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
        return spaces.Dict(
            {
                "position": self.observation_space(params),
                "terminated": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
