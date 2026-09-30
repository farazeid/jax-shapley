"""Narrow typed adapter around Gymnax's dynamic step metadata boundary."""

from typing import cast

import jax
from gymnax.environments.environment import Environment, EnvParams, EnvState

from jax_shapley.envs.contracts import Flag, Observation, Reward, StepInfo


class TypedEnvironment[State: EnvState, Params: EnvParams](Environment[State, Params]):
    """Retain Gymnax transitions and auto-reset while exposing our info schema.

    Gymnax types step metadata as dict[Any, Any]. Our concrete step_env methods
    produce StepInfo; this adapter narrows only that external library boundary.
    """

    def step(
        self,
        key: jax.Array,
        state: State,
        action: float | jax.Array,
        params: Params | None = None,
    ) -> tuple[Observation, State, Reward, Flag, StepInfo]:
        # Gymnax's decorated generic method loses its concrete return type under ty.
        return cast(tuple[Observation, State, Reward, Flag, StepInfo], super().step(key, state, action, params))

    # TypedDict is a runtime dict, but not a static subtype of Gymnax's dict[Any, Any].
    # This single library-boundary override lets concrete environments use StepInfo.
    def step_env(  # ty: ignore[invalid-method-override]
        self,
        key: jax.Array,
        state: State,
        action: float | jax.Array,
        params: Params,
    ) -> tuple[Observation, State, Reward, Flag, StepInfo]:
        raise NotImplementedError
