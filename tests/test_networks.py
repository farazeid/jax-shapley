"""Offline network and gradient parity with the pinned FastSVERL builder."""

import hashlib
import json
from collections.abc import Mapping
from math import sqrt
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.networks import MLP, LayerConfig, Linear, MLPConfig, OrthogonalInit, PPOActor, PPOCritic, QNetwork
from tests.network_cases import NETWORK_CASES, AgentNetwork, NetworkCase, make_network

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_networks.npz"


class _LinearGradient(Protocol):
    kernel: nnx.Param[Array]
    bias: nnx.Param[Array]


class _MLPGradient(Protocol):
    layers: Mapping[int, _LinearGradient]


class _AgentGradient(Protocol):
    mlp: _MLPGradient


def test_fixture_provenance() -> None:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert metadata["fastsverl_commit"] == "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
    assert metadata["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert metadata["torch_version"].startswith("2.6.0")


@pytest.mark.parametrize("case", NETWORK_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_upstream_outputs_and_gradients(case: NetworkCase, compiled: bool) -> None:
    model = make_network(case)
    with np.load(FIXTURE, allow_pickle=False) as data:
        for index, layer in enumerate(model.mlp.layers):
            layer.kernel[...] = jnp.asarray(data[f"{case.name}_kernel{index}"])
            layer.bias[...] = jnp.asarray(data[f"{case.name}_bias{index}"])
        inputs = jnp.asarray(data[f"{case.name}_inputs"])
        probe = jnp.asarray(data[f"{case.name}_probe"])
        forward = nnx.jit(lambda model, inputs: model(inputs)) if compiled else lambda model, inputs: model(inputs)
        output = forward(model, inputs)[0]
        np.testing.assert_allclose(output, data[f"{case.name}_outputs"], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(model(inputs.astype(jnp.int32))[0], output, rtol=0, atol=0)
        differentiate = nnx.value_and_grad(_probe_loss)
        if compiled:
            differentiate = nnx.jit(differentiate)
        loss, raw_gradients = differentiate(model, inputs, probe)
        # NNX's gradient State mirrors the graph, but its dynamic accessor erases types.
        gradients = cast(_AgentGradient, raw_gradients)
        np.testing.assert_allclose(loss, data[f"{case.name}_loss"], rtol=2e-5, atol=2e-6)
        assert len(jax.tree.leaves(raw_gradients)) == 2 * len(model.mlp.layers)
        for index, layer in enumerate(model.mlp.layers):
            gradient = gradients.mlp.layers[index]
            assert gradient.kernel[...].shape == layer.kernel[...].shape
            assert gradient.bias[...].shape == layer.bias[...].shape
            for key, actual in (("kernel", gradient.kernel[...]), ("bias", gradient.bias[...])):
                assert actual.dtype == jnp.float32
                assert jnp.all(jnp.isfinite(actual))
                np.testing.assert_allclose(actual, data[f"{case.name}_{key}_grad{index}"], rtol=3e-5, atol=3e-6)
        input_gradient = jax.grad(lambda observations: _probe_loss(model, observations, probe))(inputs)
        np.testing.assert_allclose(input_gradient, data[f"{case.name}_input_grad"], rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("input_width,output_width", [(3, 7), (7, 3), (4, 4)])
def test_initialization_contract(input_width: int, output_width: int) -> None:
    uniform = Linear(input_width, LayerConfig(output_width), rngs=nnx.Rngs(42))
    repeated = Linear(input_width, LayerConfig(output_width), rngs=nnx.Rngs(42))
    different = Linear(input_width, LayerConfig(output_width), rngs=nnx.Rngs(43))
    bound = 1 / sqrt(input_width)
    for values in (uniform.kernel[...], uniform.bias[...]):
        assert values.dtype == jnp.float32
        assert jnp.all((values >= -bound) & (values <= bound))
        assert jnp.any(values != 0)
    np.testing.assert_array_equal(uniform.kernel[...], repeated.kernel[...])
    assert not jnp.array_equal(uniform.kernel[...], different.kernel[...])
    orthogonal = Linear(
        input_width, LayerConfig(output_width, initialization=OrthogonalInit(1.7, 0.25)), rngs=nnx.Rngs(42)
    )
    kernel = orthogonal.kernel[...]
    gram = kernel @ kernel.T if input_width < output_width else kernel.T @ kernel
    np.testing.assert_allclose(gram, 1.7**2 * jnp.eye(min(input_width, output_width)), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(orthogonal.bias[...], jnp.full((output_width,), 0.25))


def test_default_initialization_distribution() -> None:
    layer = Linear(64, LayerConfig(256), rngs=nnx.Rngs(1729))
    # A large deterministic sample distinguishes the intended uniform law from
    # constant/zero, clipped-normal or incorrectly scaled initializers.
    kernel = np.asarray(layer.kernel[...])
    bound = 1 / sqrt(64)
    assert abs(kernel.mean()) < bound * 0.03
    np.testing.assert_allclose(kernel.var(), bound**2 / 3, rtol=0.05)


def test_known_affine_example() -> None:
    model = MLP(MLPConfig(2, (LayerConfig(1),)), rngs=nnx.Rngs(0))
    model.layers[0].kernel[...] = jnp.asarray([[2.0], [-3.0]])
    model.layers[0].bias[...] = jnp.asarray([0.5])
    np.testing.assert_array_equal(model(jnp.asarray([4, 1], jnp.int32)), jnp.asarray([5.5]))


def test_invalid_contracts_fail_before_execution() -> None:
    with pytest.raises(ValueError, match="positive"):
        LayerConfig(0)
    with pytest.raises(ValueError, match="positive"):
        MLPConfig(0, (LayerConfig(1),))
    with pytest.raises(ValueError, match="at least one"):
        MLPConfig(2, ())
    with pytest.raises(ValueError, match="finite"):
        OrthogonalInit(float("nan"))
    with pytest.raises(ValueError, match="Q-values"):
        QNetwork(MLPConfig(2, (LayerConfig(3, "softmax"),)), rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="softmax"):
        PPOActor(MLPConfig(2, (LayerConfig(3),)), rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="one unactivated"):
        PPOCritic(MLPConfig(2, (LayerConfig(2),)), rngs=nnx.Rngs(0))
    model = make_network(NETWORK_CASES[0])
    for shape in ((), (3,), (4, 3)):
        with pytest.raises(ValueError, match="final observation axis"):
            nnx.jit(lambda model, inputs: model(inputs)).trace(model, jax.ShapeDtypeStruct(shape, jnp.int32))
    for dtype in (jnp.bool_, jnp.float16, jnp.bfloat16):
        with pytest.raises(TypeError, match="int32 or float32"):
            nnx.jit(lambda model, inputs: model(inputs)).trace(model, jax.ShapeDtypeStruct((2,), dtype))


def _probe_loss(model: AgentNetwork, observations: Array, probe: Array) -> Array:
    return jnp.sum(model(observations)[0] * probe)
