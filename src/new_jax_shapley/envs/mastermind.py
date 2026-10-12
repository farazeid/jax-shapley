"""JAX port of FastSVERL's Mastermind environment for any code size, guess count and peg count.

`Mastermind(code_size, num_guesses, num_pegs)` reproduces upstream
`Mastermind(code_size, num_guesses, num_pegs)`. The paper uses 2-2-2, 3-3-3 and
4-4-3, 4-5-3, 4-6-3 (code size, guesses, pegs). `Mastermind222` keeps the
original name with the 2-2-2 defaults; every 2-2-2 random draw is unchanged.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
import tyro
from flax import struct
from gymnax.environments import environment, spaces

from new_jax_shapley.envs.base import TypedEnvironment
from new_jax_shapley.envs.contracts import Board, Code, Counter, Flag, Observation, Reward, StepInfo

if TYPE_CHECKING:
    import numpy as np

    from new_jax_shapley.envs.render import RenderConfig


@struct.dataclass
class MastermindState(environment.EnvState):
    """Complete Markov state: the public board (one row per guess) and the hidden code."""

    board: Board
    goal: Code
    guess_index: Counter
    terminated: Flag
    time: Counter


@struct.dataclass
class MastermindParams(environment.EnvParams):
    """Episode bound; the default matches 2-2-2, and `Mastermind.default_params` sets num_guesses."""

    max_steps_in_episode: int = 2


@dataclass(frozen=True, slots=True)
class Mastermind(TypedEnvironment[MastermindState, MastermindParams]):
    """FastSVERL's code-breaking game: guess a `code_size`-peg code over `num_pegs` colours in `num_guesses`.

    Each board row is `[misplaced, guess pegs..., exact]`; empty rows are -1. The
    reward is -1 per guess, plus `num_guesses` on the winning guess.
    """

    code_size: int = 2
    num_guesses: int = 2
    num_pegs: int = 2

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) is tyro.MISSING for item in fields(self)):
            return  # A sweep config: validated once the command line supplies every field.
        if self.code_size < 1 or self.num_guesses < 1 or self.num_pegs < 2:
            raise ValueError("Mastermind needs code_size >= 1, num_guesses >= 1 and num_pegs >= 2")

    @property
    def default_params(self) -> MastermindParams:
        return MastermindParams(max_steps_in_episode=self.num_guesses)

    def reset_env(
        self,
        key: jax.Array,
        params: MastermindParams,
    ) -> tuple[Observation, MastermindState]:
        del params
        goal_index = jax.random.randint(key, (), 0, self.num_actions)
        state = MastermindState(
            board=jnp.full((self.num_guesses, self.code_size + 2), -1, dtype=jnp.int32),
            goal=self.decode_action(goal_index),
            guess_index=jnp.asarray(0, dtype=jnp.int32),
            terminated=jnp.asarray(False),
            time=jnp.asarray(0, dtype=jnp.int32),
        )
        return self.get_obs(state), state

    def step_env(
        self,
        key: jax.Array,
        state: MastermindState,
        action: float | jax.Array,
        params: MastermindParams,
    ) -> tuple[Observation, MastermindState, Reward, Flag, StepInfo]:
        del key, params
        guess = self.decode_action(action)
        exact_mask = guess == state.goal
        num_exact = jnp.sum(exact_mask, dtype=jnp.int32)

        guess_one_hot = jax.nn.one_hot(guess - 1, self.num_pegs, dtype=jnp.int32)
        goal_one_hot = jax.nn.one_hot(state.goal - 1, self.num_pegs, dtype=jnp.int32)
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
        next_state = MastermindState(
            board=board,
            goal=state.goal,
            guess_index=state.guess_index + jnp.asarray(1, dtype=jnp.int32),
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
        state: MastermindState,
        params: MastermindParams | None = None,
        render_config: "RenderConfig | None" = None,
    ) -> "np.ndarray":
        """Render the public board as a host-side RGB array."""
        del params
        from new_jax_shapley.envs.render.mastermind import render_mastermind

        return render_mastermind(
            state,
            num_guesses=self.num_guesses,
            code_size=self.code_size,
            num_pegs=self.num_pegs,
            config=render_config,
        )

    @property
    def name(self) -> str:
        return f"Mastermind-{self.code_size}{self.num_guesses}{self.num_pegs}-v0"

    @property
    def num_actions(self) -> int:
        """Every code is a guess: num_pegs ** code_size."""
        return int(self.num_pegs**self.code_size)  # int ** int is untyped in typeshed.

    def action_space(self, params: MastermindParams | None = None) -> spaces.Discrete:
        del params
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: MastermindParams) -> spaces.Box:
        """Board cells: -1 (empty), pegs 1..num_pegs and clues 0..code_size."""
        del params
        shape = (self.num_guesses * (self.code_size + 2),)
        return spaces.Box(low=-1, high=self._high, shape=shape, dtype=jnp.int32)

    def state_space(self, params: MastermindParams) -> spaces.Dict:
        return spaces.Dict({
            "board": spaces.Box(low=-1, high=self._high, shape=(self.num_guesses, self.code_size + 2), dtype=jnp.int32),
            "goal": spaces.Box(low=1, high=self.num_pegs, shape=(self.code_size,), dtype=jnp.int32),
            "guess_index": spaces.Discrete(self.num_guesses + 1),
            "terminated": spaces.Discrete(2),
            "time": spaces.Discrete(params.max_steps_in_episode + 1),
        })

    @overload
    def get_obs(self, state: MastermindState) -> Observation: ...

    @overload
    def get_obs(self, state: MastermindState, params: MastermindParams) -> Observation: ...

    @overload
    def get_obs(
        self,
        state: MastermindState,
        key: jax.Array,
        params: MastermindParams,
    ) -> Observation: ...

    def get_obs(
        self,
        state: MastermindState,
        *args: MastermindParams | jax.Array,
        params: MastermindParams | None = None,
        key: jax.Array | None = None,
    ) -> Observation:
        del args, params, key
        return state.board.reshape(-1)

    def is_terminal(self, state: MastermindState, params: MastermindParams) -> Flag:
        del params
        return state.terminated

    def decode_action(self, action: float | jax.Array) -> Code:
        """Decode actions exactly like FastSVERL's least-significant-first base-num_pegs table."""
        action = jnp.asarray(action, dtype=jnp.int32)
        powers = jnp.asarray([self.num_pegs**index for index in range(self.code_size)], dtype=jnp.int32)
        return ((action // powers) % self.num_pegs + 1).astype(jnp.int32)

    @property
    def _high(self) -> int:
        return max(self.num_pegs, self.code_size)


# The original 2-2-2 names, kept for existing runs, recipes and tests.
Mastermind222 = Mastermind
Mastermind222State = MastermindState
Mastermind222Params = MastermindParams
