"""Seeds within a stage train independently, and the pipeline pairs agents, characteristics and Shapley models."""

from dataclasses import replace

import jax
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.supervised_learning import SupervisedConfig, seed_slice
from jax_shapley.envs import Mastermind
from jax_shapley.experiment import ExperimentConfig
from jax_shapley.main import run
from jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(2, 2, 2)
TRAINING = SupervisedConfig(epochs=3, batch_size=8, buffer_size=40, num_envs=4)
AGENT = DQNConfig(
    env=ENV,
    hidden_widths=(8,),
    num_envs=4,
    total_timesteps=256,
    buffer_size=256,
    batch_size=8,
    learning_starts=32,
    train_frequency=4,
)
CHARACTERISTIC = CharacteristicConfig(ENV, hidden_widths=(8,), seeds=(1, 2, 3), train=TRAINING)
SHAPLEY = ShapleyConfig(ENV, hidden_widths=(8,), buffers="characteristic", seeds=(1, 2, 3), train=TRAINING)


def test_stage_seeds_equal_single_seed_stages() -> None:
    result = run(ExperimentConfig(name="pairing", algorithm=AGENT, characteristic=CHARACTERISTIC, shapley=()))
    swept, swept_metrics = result.characteristic_state, result.characteristic_metrics
    assert swept_metrics is not None
    for index, seed in enumerate(CHARACTERISTIC.seeds):
        single = run(
            ExperimentConfig(
                name="pairing",
                algorithm=AGENT,
                characteristic=replace(CHARACTERISTIC, seeds=(seed,)),
                shapley=(),
                explained_seed=seed,
            )
        )
        alone, metrics = seed_slice(single.characteristic_state, 0), single.characteristic_metrics
        assert metrics is not None
        one = seed_slice(swept, index)
        for name in ("train", "validation", "exact"):
            np.testing.assert_array_equal(getattr(one.buffers, name), getattr(alone.buffers, name))
        assert int(one.epoch) == int(alone.epoch) and int(one.best_epoch) == int(alone.best_epoch)
        for name in ("train_loss", "validation_loss", "exact_loss"):
            np.testing.assert_allclose(
                np.asarray(getattr(swept_metrics, name))[:, index],
                np.asarray(getattr(metrics, name))[:, 0],
                rtol=1e-5,
                atol=1e-7,
            )
        for left, right in zip(
            jax.tree.leaves(nnx.state(one.model)), jax.tree.leaves(nnx.state(alone.model)), strict=True
        ):
            np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("explained_seed", [None, 2])
def test_shapley_seeds_explain_their_characteristic_seed(explained_seed: int | None) -> None:
    """Shapley stages on the characteristic's buffers show which characteristic seed each explains."""
    config = ExperimentConfig(
        name="pairing",
        algorithm=AGENT,
        characteristic=CHARACTERISTIC,
        shapley=(SHAPLEY,),
        explained_seed=explained_seed,
    )
    result = run(config)
    shapley_state = result.shapley[0][0]
    for index in range(len(SHAPLEY.seeds)):
        source = index if explained_seed is None else CHARACTERISTIC.seeds.index(explained_seed)
        np.testing.assert_array_equal(
            seed_slice(shapley_state, index).buffers.train,
            seed_slice(result.characteristic_state, source).buffers.train,
        )


def test_per_seed_agents_pair_with_characteristic_seeds() -> None:
    one = run(ExperimentConfig(name="pairing", algorithm=AGENT, characteristic=CHARACTERISTIC, shapley=()))
    many = run(
        ExperimentConfig(
            name="pairing", algorithm=replace(AGENT, seeds=(1, 2, 3)), characteristic=CHARACTERISTIC, shapley=()
        )
    )
    first, other = one.characteristic_state.buffers.train, many.characteristic_state.buffers.train
    np.testing.assert_array_equal(first[0], other[0])  # Seed 1 explains agent 1 either way.
    assert not np.array_equal(first[1], other[1])  # Seed 2 explains agent 2, not agent 1.
    with pytest.raises(ValueError, match="one per characteristic seed"):
        ExperimentConfig(
            name="pairing", algorithm=replace(AGENT, seeds=(1, 2)), characteristic=CHARACTERISTIC, shapley=()
        )
    with pytest.raises(ValueError, match="characteristic seeds"):
        ExperimentConfig(
            name="pairing",
            algorithm=AGENT,
            characteristic=CHARACTERISTIC,
            shapley=(replace(SHAPLEY, seeds=(1, 2)),),
            explained_seed=None,
        )
