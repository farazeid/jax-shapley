"""Tensor and metadata contracts shared by the FastSVERL environment ports.

Leading axes document the batches introduced by vmap/scan; each raw environment
step consumes one state and produces scalar transition metadata. Shape annotations
do not install runtime checks or replace compilation and numerical tests.
"""

from typing import TypedDict

from jax import Array
from jaxtyping import Bool, Float, Int

# Ruff interprets jaxtyping dimension strings as Python forward references.
# These targeted suppressions apply only to non-Python shape grammar.
type Observation = Int[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error]
type Position2 = Int[Array, "... 2"]  # ruff: ignore[forward-annotation-syntax-error]
type Board222 = Int[Array, "... 2 4"]  # ruff: ignore[forward-annotation-syntax-error]
type Counter = Int[Array, "..."]
type Flag = Bool[Array, "..."]
type Reward = Float[Array, "..."]


class StepInfo(TypedDict):
    """Metadata from a raw transition, retained across Gymnax auto-reset."""

    discount: Reward
    terminated: Flag
    truncated: Flag


EpisodeInfo = TypedDict("EpisodeInfo", {"return": Reward, "length": Counter, "completed": Flag})


class EpisodeStepInfo(StepInfo):
    """Transition metadata plus the wrapper's masked episode event."""

    episode: EpisodeInfo
