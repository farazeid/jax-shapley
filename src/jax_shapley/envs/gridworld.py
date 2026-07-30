"""JAX port of FastSVERL's deterministic 2 x 4 Gridworld."""

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@struct.dataclass
class GridworldState(environment.EnvState):
    """Gridworld state carried through JAX transformations."""

    position: jax.Array
    terminated: jax.Array
    time: jax.Array


@struct.dataclass
class GridworldParams(environment.EnvParams):
    """Gridworld parameters; the source environment has no time limit."""

    max_steps_in_episode: int = 2**31 - 1


class Gridworld(environment.Environment[GridworldState, GridworldParams]):
    """FastSVERL-compatible Gridworld (registered there as ``GWB-v0``)."""

    _action_deltas = jnp.asarray([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32)
    _starts = jnp.asarray([[0, 0], [1, 0]], dtype=jnp.int32)
    _blocked = jnp.asarray([0, 1], dtype=jnp.int32)

    @property
    def default_params(self) -> GridworldParams:
        return GridworldParams()

    def reset_env(
        self, key: jax.Array, params: GridworldParams
    ) -> tuple[jax.Array, GridworldState]:
        del params
        start_index = jax.random.randint(key, (), 0, 2)
        position = self._starts[start_index]
        state = GridworldState(
            position=position,
            terminated=jnp.asarray(False),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def step_env(
        self,
        key: jax.Array,
        state: GridworldState,
        action: int | jax.Array,
        params: GridworldParams,
    ) -> tuple[jax.Array, GridworldState, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, params
        candidate = state.position + self._action_deltas[action]
        in_bounds = (
            (candidate[0] >= 0) & (candidate[0] < 2) & (candidate[1] >= 0) & (candidate[1] < 4)
        )
        not_blocked = jnp.any(candidate != self._blocked)
        moved_position = jnp.where(in_bounds & not_blocked, candidate, state.position)
        terminated = moved_position[1] == 3

        # FastSVERL deliberately maps terminal transitions back to [0, 0].
        position = jnp.where(terminated, jnp.zeros(2, dtype=jnp.int32), moved_position)
        reward = jnp.where(terminated, 9.0, -1.0).astype(jnp.float32)
        next_state = GridworldState(
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
        state: GridworldState,
        params: GridworldParams | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del params, key
        return state.position

    def is_terminal(self, state: GridworldState, params: GridworldParams) -> jax.Array:
        del params
        return state.terminated

    @property
    def name(self) -> str:
        return "GWB-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: GridworldParams | None = None) -> spaces.Discrete:
        del params
        return spaces.Discrete(4)

    def observation_space(self, params: GridworldParams) -> spaces.Box:
        del params
        return spaces.Box(
            low=jnp.asarray([0, 0], dtype=jnp.int32),
            high=jnp.asarray([1, 3], dtype=jnp.int32),
            shape=(2,),
            dtype=jnp.int32,
        )

    def state_space(self, params: GridworldParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "position": self.observation_space(params),
                "terminated": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )
