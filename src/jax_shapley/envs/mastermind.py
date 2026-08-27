"""JAX port of FastSVERL's Mastermind-222 environment."""

from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

if TYPE_CHECKING:
    import numpy as np

    from jax_shapley.envs.render import RenderConfig


@struct.dataclass
class Mastermind222State(environment.EnvState):
    """Complete Markov state for Mastermind-222."""

    board: jax.Array
    goal: jax.Array
    guess_index: jax.Array
    terminated: jax.Array
    time: jax.Array


@struct.dataclass
class Mastermind222Params(environment.EnvParams):
    """Fixed parameters for the 2-code, 2-guess, 2-peg domain."""

    max_steps_in_episode: int = 2


class Mastermind222(environment.Environment[Mastermind222State, Mastermind222Params]):
    """The Mastermind-222 configuration used by FastSVERL."""

    code_size = 2
    num_guesses = 2
    num_pegs = 2

    @staticmethod
    def decode_action(action: float | jax.Array) -> jax.Array:
        """Decode actions exactly like FastSVERL's least-significant-first table."""
        action = jnp.asarray(action, dtype=jnp.int32)
        powers = jnp.asarray([1, 2], dtype=jnp.int32)
        return ((action // powers) % 2 + 1).astype(jnp.int32)

    @property
    def default_params(self) -> Mastermind222Params:
        return Mastermind222Params()

    def reset_env(
        self,
        key: jax.Array,
        params: Mastermind222Params,
    ) -> tuple[jax.Array, Mastermind222State]:
        del params
        goal_index = jax.random.randint(key, (), 0, 4)
        state = Mastermind222State(
            board=jnp.full((2, 4), -1, dtype=jnp.int32),
            goal=self.decode_action(goal_index),
            guess_index=jnp.asarray(0, dtype=jnp.int32),
            terminated=jnp.asarray(False),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def step_env(
        self,
        key: jax.Array,
        state: Mastermind222State,
        action: float | jax.Array,
        params: Mastermind222Params,
    ) -> tuple[jax.Array, Mastermind222State, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key, params
        guess = self.decode_action(action)
        exact_mask = guess == state.goal
        num_exact = jnp.sum(exact_mask, dtype=jnp.int32)

        guess_one_hot = jax.nn.one_hot(guess - 1, 2, dtype=jnp.int32)
        goal_one_hot = jax.nn.one_hot(state.goal - 1, 2, dtype=jnp.int32)
        unmatched = (~exact_mask).astype(jnp.int32)[:, None]
        guess_counts = jnp.sum(guess_one_hot * unmatched, axis=0)
        goal_counts = jnp.sum(goal_one_hot * unmatched, axis=0)
        num_misplaced = jnp.sum(jnp.minimum(guess_counts, goal_counts), dtype=jnp.int32)

        row = jnp.concatenate((
            jnp.asarray([num_misplaced], dtype=jnp.int32),
            guess,
            jnp.asarray([num_exact], dtype=jnp.int32),
        ))
        board = state.board.at[state.guess_index].set(row)
        correct = num_exact == self.code_size
        terminated = correct | (state.guess_index == self.num_guesses - 1)
        reward = (-1.0 + jnp.where(correct, float(self.num_guesses), 0.0)).astype(jnp.float32)
        next_state = Mastermind222State(
            board=board,
            goal=state.goal,
            guess_index=state.guess_index + jnp.asarray(1, dtype=jnp.int32),
            terminated=terminated,
            time=state.time + jnp.asarray(1, dtype=jnp.int32),
        )
        info: dict[str, jax.Array] = {
            "discount": jnp.where(terminated, 0.0, 1.0).astype(jnp.float32),
            "terminated": terminated,
            "truncated": jnp.asarray(False),
        }
        return self.get_obs(next_state), next_state, reward, terminated, info

    @overload
    def get_obs(self, state: Mastermind222State) -> jax.Array: ...

    @overload
    def get_obs(self, state: Mastermind222State, params: Mastermind222Params) -> jax.Array: ...

    @overload
    def get_obs(
        self,
        state: Mastermind222State,
        key: jax.Array,
        params: Mastermind222Params,
    ) -> jax.Array: ...

    def get_obs(
        self,
        state: Mastermind222State,
        *args: Mastermind222Params | jax.Array,
        params: Mastermind222Params | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del args, params, key
        return state.board.reshape(-1)

    def is_terminal(self, state: Mastermind222State, params: Mastermind222Params) -> jax.Array:
        del params
        return state.terminated

    @property
    def name(self) -> str:
        return "Mastermind-222-v0"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: Mastermind222Params | None = None) -> spaces.Discrete:
        del params
        return spaces.Discrete(4)

    def observation_space(self, params: Mastermind222Params) -> spaces.Box:
        del params
        return spaces.Box(low=-1, high=2, shape=(8,), dtype=jnp.int32)

    def render(
        self,
        state: Mastermind222State,
        params: Mastermind222Params | None = None,
        render_config: "RenderConfig | None" = None,
    ) -> "np.ndarray":
        """Render the public board as a host-side RGB array."""
        del params
        from jax_shapley.envs.render.mastermind import render_mastermind

        return render_mastermind(
            state,
            num_guesses=self.num_guesses,
            code_size=self.code_size,
            config=render_config,
        )

    def state_space(self, params: Mastermind222Params) -> spaces.Dict:
        return spaces.Dict({
            "board": spaces.Box(low=-1, high=2, shape=(2, 4), dtype=jnp.int32),
            "goal": spaces.Box(low=1, high=2, shape=(2,), dtype=jnp.int32),
            "guess_index": spaces.Discrete(3),
            "terminated": spaces.Discrete(2),
            "time": spaces.Discrete(params.max_steps_in_episode + 1),
        })
