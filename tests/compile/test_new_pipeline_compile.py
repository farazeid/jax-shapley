"""new_jax_shapley's stage programs traced, lowered and compiled with live-logging callbacks.

Shapes are the registered Mastermind-222 behaviour experiment's (20 seeds, 10,000-row
buffers, 100,000 agent steps over 16 environments); nothing executes.
"""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from new_jax_shapley.algorithms import dqn
from new_jax_shapley.characteristic_models import behaviour as characteristic
from new_jax_shapley.common.supervised_learning import EpochMetrics, seed_slice, stack_seeds
from new_jax_shapley.experiment import load_configs
from new_jax_shapley.shapley_models import behaviour as shapley

KEY = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)
# Mastermind-222 has 53 states: the exact tables' size for its buffers.
MAX_UNIQUE = 53


@pytest.mark.compile
def test_tracked_stage_programs_compile() -> None:
    _, config = load_configs()["mastermind222_behaviour"]
    algorithm = config.algorithm
    nnx.jit(partial(dqn.train, algorithm, _ignore_step, _ignore_evaluation)).trace(
        KEY, jax.ShapeDtypeStruct((), jnp.int32)
    ).lower().compile()

    stage = config.characteristic
    seeds = len(stage.seeds)
    keys = jax.ShapeDtypeStruct((seeds,), KEY.dtype)
    agent = dqn.trained_agent(algorithm, dqn.initialize(algorithm, jax.random.key(0)))
    agents = stack_seeds([agent] * seeds)
    template = characteristic.initialize(stage)
    nnx.jit(partial(characteristic.collect, stage)).trace(keys, agents).lower().compile()
    training = nnx.jit(partial(characteristic.fit, stage, MAX_UNIQUE, _ignore_epoch))
    training.trace(keys, agents, template.buffers).lower().compile()

    for shapley_config in config.shapley:
        shapley_seeds = len(shapley_config.seeds)
        explained = seed_slice(template, 0)
        program = nnx.jit(partial(shapley.fit, shapley_config, stage, MAX_UNIQUE, _ignore_epoch))
        program.trace(
            jax.ShapeDtypeStruct((shapley_seeds,), KEY.dtype),
            stack_seeds([agent] * shapley_seeds),
            stack_seeds([explained.model] * shapley_seeds),
            stack_seeds([explained.buffers] * shapley_seeds),
        ).lower().compile()


def _ignore_step(seed: jax.Array, metrics: dqn.StepMetrics) -> None:
    del seed, metrics


def _ignore_evaluation(seed: jax.Array, metrics: dqn.EvaluationMetrics) -> None:
    del seed, metrics


def _ignore_epoch(metrics: EpochMetrics) -> None:
    del metrics
