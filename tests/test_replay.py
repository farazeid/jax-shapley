"""Known insertion examples and replacement sampling independent of Torch."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley.replay import TransitionBatch, append, empty_replay, gather, sample


def test_partial_wraparound_and_oversized_insertion() -> None:
    replay = append(empty_replay(5, 2), batch(0, 3))
    assert (int(replay.pointer), int(replay.size)) == (3, 3)
    replay = append(replay, batch(3, 4))
    np.testing.assert_array_equal(replay.data.actions, [5, 6, 2, 3, 4])
    assert (int(replay.pointer), int(replay.size)) == (2, 5)
    # Upstream stores only the last capacity items, starting at the OLD pointer.
    replay = append(replay, batch(7, 8))
    np.testing.assert_array_equal(replay.data.actions, [13, 14, 10, 11, 12])
    assert (int(replay.pointer), int(replay.size)) == (2, 5)
    selected = gather(replay, jnp.array([2, 2, 0], jnp.int32))
    np.testing.assert_array_equal(selected.actions, [10, 10, 13])
    np.testing.assert_array_equal(selected.next_observations, selected.observations + 1)
    selected.validate()


def test_sampling_replacement_nonfull_and_explicit_keys() -> None:
    replay = append(empty_replay(8, 2), batch(4, 2))
    key = jax.random.key(1)
    result, next_key = sample(replay, 64, key)
    repeated, repeated_key = sample(replay, 64, key)
    np.testing.assert_array_equal(result.actions, repeated.actions)
    np.testing.assert_array_equal(jax.random.key_data(next_key), jax.random.key_data(repeated_key))
    assert set(np.asarray(result.actions).tolist()) == {4, 5}
    assert not jnp.array_equal(jax.random.key_data(key), jax.random.key_data(next_key))
    assert result.actions.shape == (64,)


def test_storage_contract_errors_and_dtypes() -> None:
    replay = empty_replay(4, 2)
    replay.data.validate()
    assert replay.pointer.dtype == replay.size.dtype == jnp.int32
    with pytest.raises(ValueError):
        empty_replay(0, 2)
    with pytest.raises(ValueError):
        append(replay, batch(0, 2, features=3))
    with pytest.raises(TypeError):
        append(replay, replace(batch(0, 2), actions=jnp.zeros(2, jnp.float32)))
    with pytest.raises(ValueError):
        append(replay, replace(batch(0, 2), rewards=jnp.zeros((2, 1), jnp.float32)))
    with pytest.raises(TypeError):
        gather(replay, jnp.zeros(1, jnp.float32))


def batch(start: int, count: int, features: int = 2) -> TransitionBatch:
    ids = jnp.arange(start, start + count, dtype=jnp.int32)
    observations = jnp.broadcast_to(ids[:, None], (count, features))
    return TransitionBatch(
        observations, ids, ids.astype(jnp.float32), observations + 1, ids % 2 == 0, -ids.astype(jnp.float32)
    )
