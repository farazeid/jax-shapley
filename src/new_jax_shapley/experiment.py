"""The experiment config: everything one run of the pipeline needs, defined in `configs/NAME.py`."""

import importlib
import pkgutil
from dataclasses import dataclass

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.backends import Track
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig


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
    explained_seed: int = 1
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
        if self.explained_seed not in self.characteristic.seeds:
            raise ValueError("explained_seed must be one of the characteristic seeds")


def load_configs() -> dict[str, tuple[str, ExperimentConfig]]:
    """Every config module under `new_jax_shapley.configs`: its name, docstring's first line and CONFIG.

    This is the mapping `tyro.extras.overridable_config_cli` takes.
    """
    package = importlib.import_module("new_jax_shapley.configs")
    configs: dict[str, tuple[str, ExperimentConfig]] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        config = getattr(module, "CONFIG", None)
        if not isinstance(config, ExperimentConfig):
            raise TypeError(f"{module.__name__} must define CONFIG = ExperimentConfig(...)")
        if config.name != module_info.name:
            raise ValueError(f"{module.__name__}: CONFIG.name must be {module_info.name!r}")
        description = (module.__doc__ or "").strip().splitlines()
        configs[module_info.name] = (description[0] if description else "", config)
    return configs
