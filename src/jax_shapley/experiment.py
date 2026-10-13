"""Experiment configs, defined in `configs/NAME.py`: the linear pipeline's (`main`) and continual learning's (`continual`)."""

import importlib
import pkgutil
from dataclasses import dataclass, field, fields
from math import inf, isfinite, isnan
from typing import Literal

import tyro

from jax_shapley.algorithms import dqn
from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.backends import Track
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.characteristic_models.outcome import OutcomeConfig
from jax_shapley.common.networks import AdamConfig
from jax_shapley.shapley_models.behaviour import ShapleyConfig

type ContinualKind = Literal["behaviour", "prediction", "outcome"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One explanation experiment. Every Shapley run explains characteristic seed `explained_seed`.

    The tracking settings (`track`, `project`, `entity`) are the experiment's: `main` copies
    them, the experiment `name` and a new run group `NAME-TIMESTAMP` into every stage.
    """

    name: str
    algorithm: DQNConfig
    characteristic: CharacteristicConfig
    shapley: tuple[ShapleyConfig, ...]
    outcome: OutcomeConfig | None = None
    """An outcome characteristic stage after the (behaviour) characteristic one; the Shapley models
    then explain outcome seed `explained_seed`."""
    explained_seed: int | None = 1
    """The characteristic seed every Shapley seed explains; None pairs Shapley seed s with characteristic seed s."""
    compile_only: bool = False
    """Only trace, lower and compile every stage (before a long run); nothing runs or is tracked."""
    track: Track = "trackio"
    project: str = "dev-jax"
    entity: str | None = "fastsverl"
    """wandb entity (team or user)."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiments need a name")
        if any(stage.env != self.algorithm.env for stage in (self.characteristic, *self.shapley)):
            raise ValueError("every stage must use the same env")
        if len(self.algorithm.seeds) not in (1, len(self.characteristic.seeds)):
            raise ValueError("the agent stage trains one agent, or one per characteristic seed")
        if self.outcome is not None and (
            self.characteristic.kind != "behaviour"
            or self.outcome.env != self.algorithm.env
            or (self.explained_seed is not None and self.explained_seed not in self.outcome.seeds)
            or (self.explained_seed is None and (self.outcome.seeds != self.characteristic.seeds or self.shapley))
        ):
            raise ValueError(
                "an outcome stage conditions on behaviour seed explained_seed (also an outcome seed, which "
                "Shapley models explain), or pairs outcome seed s with behaviour seed s (no Shapley models)"
            )
        if self.explained_seed is None:
            if any(stage.seeds != self.characteristic.seeds for stage in self.shapley):
                raise ValueError("paired Shapley seeds must equal the characteristic seeds")
        elif self.explained_seed not in self.characteristic.seeds:
            raise ValueError("explained_seed must be one of the characteristic seeds")


@dataclass(frozen=True, slots=True)
class ContinualExplainConfig:
    """Explanation-side settings of the source's parallel drivers.

    The explained policy is the agent's explore policy. `with_exact_char` selects the Shapley
    ground truth (exact characteristic: the w_exact and w_policy groups);
    `outcome_with_exact_char` conditions exact outcome values on the exact behaviour
    characteristic (w_policy). Outcome characteristics are off-policy.
    """

    kind: ContinualKind = "behaviour"
    update_ratio: int = 1
    with_exact_char: bool = True
    outcome_with_exact_char: bool = True
    exact_frequency: int = 1_000
    exact_buffer_size: int = 10_000
    hidden_widths: tuple[int, ...] = (64, 64)
    learning_rate: float = 2.5e-4
    tau: float = 0.01
    gamma: float = 1.0
    weighting: bool = True
    clip_coef: float = inf
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) is tyro.MISSING for item in fields(self)):
            return  # A sweep config: validated once the command line supplies every field.
        if self.kind not in ("behaviour", "prediction", "outcome"):
            raise ValueError("kind must be behaviour, prediction or outcome")
        for name, value in (
            ("update_ratio", self.update_ratio),
            ("exact_frequency", self.exact_frequency),
            ("exact_buffer_size", self.exact_buffer_size),
            ("hidden widths", min(self.hidden_widths, default=0)),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0 or isnan(self.clip_coef) or self.clip_coef < 0:
            raise ValueError("learning_rate must be positive and clip_coef nonnegative")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("tau must be in (0, 1] and gamma in [0, 1]")


@dataclass(frozen=True, slots=True)
class ContinualConfig:
    """A continual experiment: one agent (`algorithm.seeds`) and its explanation models per seed."""

    name: str
    algorithm: dqn.DQNConfig
    explain: ContinualExplainConfig = field(default_factory=ContinualExplainConfig)
    checkpoint: str | None = None
    """A continual artifact to load instead of training, e.g. `GROUP--continual--behaviour:latest`."""
    compile_only: bool = False
    """Only trace, lower and compile the run (before a long one); nothing runs or is tracked."""
    track: Track = "trackio"
    project: str = "dev-jax"
    entity: str | None = "fastsverl"
    """wandb entity (team or user)."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiments need a name")
        explain, num_envs = self.explain, self.algorithm.num_envs
        complete = explain.exact_buffer_size is not tyro.MISSING and num_envs is not tyro.MISSING
        if complete and explain.exact_buffer_size % num_envs:
            raise ValueError("exact_buffer_size must be a multiple of num_envs")


def load_configs[Config](config_type: type[Config]) -> dict[str, tuple[str, Config]]:
    """Every config module under `jax_shapley.configs` defining a `config_type` CONFIG: its name,
    docstring's first line and CONFIG (pipeline experiments, or continual ones).

    This is the mapping `tyro.extras.overridable_config_cli` takes.
    """
    package = importlib.import_module("jax_shapley.configs")
    configs: dict[str, tuple[str, Config]] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        config = getattr(module, "CONFIG", None)
        if config is None:
            raise TypeError(f"{module.__name__} must define CONFIG")
        if not isinstance(config, config_type):
            continue
        if getattr(config, "name", None) != module_info.name:
            raise ValueError(f"{module.__name__}: CONFIG.name must be {module_info.name!r}")
        description = (module.__doc__ or "").strip().splitlines()
        configs[module_info.name] = (description[0] if description else "", config)
    return configs
