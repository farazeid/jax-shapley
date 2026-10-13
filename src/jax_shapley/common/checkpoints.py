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
SCHEMA = 1


def save(directory: Path, state: object, settings: Mapping[str, object]) -> None:
    """Write `state` and its settings into a new folder, the content of one artifact."""
    directory.mkdir(parents=True)
    _, tree = nnx.split(state)
    leaves, structure = jax.tree.flatten(nnx.to_pure_dict(tree))
    stored = [
        jax.random.key_data(leaf) if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf for leaf in leaves
    ]
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save((directory / STATE).resolve(), jax.tree.unflatten(structure, stored))
    checkpointer.wait_until_finished()
    (directory / METADATA).write_text(json.dumps({"schema": SCHEMA, **settings}, indent=2, default=str))


def load[State](directory: Path, template: State, settings: Mapping[str, object]) -> State:
    """Restore a state saved by `save`; `settings` must equal the saved ones."""
    saved = json.loads((directory / METADATA).read_text())
    expected = json.loads(json.dumps({"schema": SCHEMA, **settings}, default=str))
    if saved != expected:
        raise ValueError(f"checkpoint {directory} was saved with different settings")
    graph, tree = nnx.split(template)
    leaves, structure = jax.tree.flatten(nnx.to_pure_dict(tree))
    keys = [jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key) for leaf in leaves]
    shapes: list[jax.ShapeDtypeStruct] = []
    for leaf, is_key in zip(leaves, keys, strict=True):
        data: jax.ShapeDtypeStruct = jax.eval_shape(jax.random.key_data, leaf) if is_key else leaf
        shapes.append(jax.ShapeDtypeStruct(data.shape, data.dtype))
    restored = ocp.StandardCheckpointer().restore((directory / STATE).resolve(), jax.tree.unflatten(structure, shapes))
    values: list[jax.Array] = []
    for value, leaf, is_key in zip(jax.tree.leaves(restored), leaves, keys, strict=True):
        wrapped: jax.Array = jax.random.wrap_key_data(value, impl=jax.random.key_impl(leaf)) if is_key else value
        values.append(wrapped)
    nnx.replace_by_pure_dict(tree, jax.tree.unflatten(structure, values))
    return nnx.merge(graph, tree)
