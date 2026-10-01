"""Validate saved external JSON once and reconstruct immutable experiment settings."""

import json
from dataclasses import fields
from pathlib import Path
from typing import cast

from jax_shapley.artifacts import describe
from jax_shapley.config import (
    AdamConfig,
    CheckpointMode,
    DQNRunConfig,
    DQNTrainConfig,
    EnvironmentConfig,
    GridworldConfig,
    HypercubeConfig,
    Mastermind222Config,
)


def load_settings(path: Path) -> DQNRunConfig[EnvironmentConfig]:
    return parse_settings(path.read_text())


def parse_settings(text: str) -> DQNRunConfig[EnvironmentConfig]:
    envelope = _mapping(json.loads(text))
    config = _mapping(envelope["config"])
    if set(config) != {field.name for field in fields(DQNRunConfig)}:
        raise ValueError("saved run settings have an invalid schema")
    train = _training(_mapping(config["train"]))
    environment = _mapping(config["env"])
    name = _text(envelope, "environment")
    env: EnvironmentConfig
    if name == "GridworldConfig" and not environment:
        env = GridworldConfig()
    elif name == "Mastermind222Config" and not environment:
        env = Mastermind222Config()
    elif name == "HypercubeConfig" and set(environment) == {"side_length", "dimension"}:
        env = HypercubeConfig(_integer(environment, "side_length"), _integer(environment, "dimension"))
    else:
        raise ValueError("unsupported saved environment")
    widths = config["hidden_widths"]
    if not isinstance(widths, list) or any(type(width) is not int for width in widths):
        raise ValueError("invalid saved hidden widths")
    mode = _text(config, "checkpoint_mode")
    if mode not in ("none", "agent", "agent-replay", "run"):
        raise ValueError("invalid saved checkpoint mode")
    result = DQNRunConfig(
        env,
        train,
        tuple(cast(list[int], widths)),
        _integer(config, "seed"),
        _integer(config, "chunk_steps"),
        Path(_text(config, "output")),
        cast(CheckpointMode, mode),
        _boolean(config, "validate_only"),
        _boolean(config, "compile_only"),
    )
    if result.validate_only or result.compile_only or json.loads(describe(result)) != envelope:
        raise ValueError("saved settings or network/reference metadata are inconsistent")
    return result


def _training(value: dict[str, object]) -> DQNTrainConfig:
    if set(value) != {field.name for field in fields(DQNTrainConfig)}:
        raise ValueError("invalid saved training schema")
    adam = _mapping(value["adam"])
    if set(adam) != {field.name for field in fields(AdamConfig)}:
        raise ValueError("invalid saved Adam schema")
    return DQNTrainConfig(
        num_envs=_integer(value, "num_envs"),
        total_timesteps=_integer(value, "total_timesteps"),
        learning_rate=_number(value, "learning_rate"),
        buffer_size=_integer(value, "buffer_size"),
        gamma=_number(value, "gamma"),
        tau=_number(value, "tau"),
        target_network_frequency=_integer(value, "target_network_frequency"),
        batch_size=_integer(value, "batch_size"),
        start_e=_number(value, "start_e"),
        end_e=_number(value, "end_e"),
        exploration_fraction=_number(value, "exploration_fraction"),
        learning_starts=_integer(value, "learning_starts"),
        train_frequency=_integer(value, "train_frequency"),
        eval_episodes=_integer(value, "eval_episodes"),
        adam=AdamConfig(_number(adam, "beta1"), _number(adam, "beta2"), _number(adam, "epsilon")),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], value)  # Validated JSON boundary; fields narrowed before constructing configs.


def _integer(value: dict[str, object], name: str) -> int:
    field = value[name]
    if type(field) is not int:
        raise ValueError(f"invalid integer setting {name}")
    return field


def _number(value: dict[str, object], name: str) -> float:
    field = value[name]
    if type(field) not in (float, int):
        raise ValueError(f"invalid numerical setting {name}")
    return cast(float, field)


def _text(value: dict[str, object], name: str) -> str:
    field = value[name]
    if not isinstance(field, str):
        raise ValueError(f"invalid text setting {name}")
    return field


def _boolean(value: dict[str, object], name: str) -> bool:
    field = value[name]
    if not isinstance(field, bool):
        raise ValueError(f"invalid boolean setting {name}")
    return field
