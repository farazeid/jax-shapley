"""Stage checkpoints: a stage's whole training state with Orbax, and its settings as JSON.

The state is any pytree of NNX graph nodes and arrays (a DQN `RunState`, a seed-stacked
`TrainState`): models, optimizers, buffers, replay, counters and keys. Typed PRNG keys are
stored as their raw key data. Restoring needs a template with the same structure, built
from the stage config exactly as the stage initialises; the saved settings are checked
against the caller's before anything is restored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from flax import nnx

STATE = "state"
METADATA = "metadata.json"
SCHEMA = 2


def save(directory: Path, state: object, settings: Mapping[str, object]) -> None:
    """Write `state` and its settings into a new folder, the content of one artifact.

    Each key's PRNG implementation is recorded, so it restores as saved whatever the
    loading process's default implementation is.
    """
    directory.mkdir(parents=True)
    _, tree = nnx.split(state)
    leaves, structure = jax.tree.flatten(nnx.to_pure_dict(tree))
    is_key = [jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) for leaf in leaves]
    stored = [jax.random.key_data(leaf) if key else leaf for leaf, key in zip(leaves, is_key, strict=True)]
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save((directory / STATE).resolve(), jax.tree.unflatten(structure, stored))
    checkpointer.wait_until_finished()
    implementations = [str(jax.random.key_impl(leaf)) for leaf, key in zip(leaves, is_key, strict=True) if key]
    metadata = {"schema": SCHEMA, "keys": implementations, "settings": settings}
    (directory / METADATA).write_text(json.dumps(metadata, indent=2, default=str))


def load[State](directory: Path, template: State, settings: Mapping[str, object]) -> State:
    """Restore a state saved by `save` onto the template's devices; `settings` must equal the saved ones."""
    saved = json.loads((directory / METADATA).read_text())
    if saved.get("schema") != SCHEMA or saved["settings"] != json.loads(json.dumps(settings, default=str)):
        raise ValueError(f"checkpoint {directory} was saved with different settings")
    graph, tree = nnx.split(template)
    leaves, structure = jax.tree.flatten(nnx.to_pure_dict(tree))
    implementations = iter(saved["keys"])
    targets: list[jax.ShapeDtypeStruct] = []
    restored_keys: list[str | None] = []
    for leaf in leaves:
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            implementation = next(implementations)
            restored_keys.append(implementation)
            # Raw key data: the key's shape, then the saved implementation's data shape.
            data = jax.random.key_data(jax.random.key(0, impl=implementation))
            sharding = jax.random.key_data(leaf).sharding
            targets.append(jax.ShapeDtypeStruct((*leaf.shape, *data.shape), data.dtype, sharding=sharding))
        else:
            restored_keys.append(None)
            targets.append(jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding))
    restored = ocp.StandardCheckpointer().restore((directory / STATE).resolve(), jax.tree.unflatten(structure, targets))
    values: list[jax.Array] = []
    for value, implementation in zip(jax.tree.leaves(restored), restored_keys, strict=True):
        wrapped: jax.Array = value if implementation is None else jax.random.wrap_key_data(value, impl=implementation)
        values.append(wrapped)
    nnx.replace_by_pure_dict(tree, jax.tree.unflatten(structure, values))
    return nnx.merge(graph, tree)
