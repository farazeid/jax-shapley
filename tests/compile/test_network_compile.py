"""Abstract NNX compilation gates, independent of training and host side effects."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from tests.network_cases import NETWORK_CASES, AgentNetwork, NetworkCase, make_network


@pytest.mark.compile
@pytest.mark.parametrize("case", NETWORK_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("leading_shape", [(), (3,), (2, 3)], ids=["single", "batch", "time-batch"])
@pytest.mark.parametrize("floating", [False, True], ids=["int32", "float32"])
def test_network_forward_and_gradient_compile(
    case: NetworkCase, leading_shape: tuple[int, ...], floating: bool
) -> None:
    _assert_compile(case, leading_shape, floating=floating)


@pytest.mark.compile
def test_network_vmap_and_scan_compile() -> None:
    model = make_network(NETWORK_CASES[1])
    spec = jax.ShapeDtypeStruct((2, 3, 8), jnp.int32)

    def scan(model: AgentNetwork, observations: Array) -> Array:
        def step(model: AgentNetwork, batch: Array) -> tuple[AgentNetwork, Array]:
            outputs = nnx.vmap(_forward, in_axes=(None, 0))(model, batch)
            return model, outputs

        _, outputs = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(model, observations)
        return outputs

    compiled = nnx.jit(scan).trace(model, spec).lower().compile()
    observations = jnp.zeros(spec.shape, dtype=jnp.int32)
    np.testing.assert_allclose(compiled(model, observations), _forward(model, observations), rtol=1e-6, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.compile
def test_network_gpu_compile() -> None:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("network GPU compilation requires an available JAX GPU backend")
    with jax.default_device(devices[0]):
        for case in NETWORK_CASES:
            for floating in (False, True):
                _assert_compile(case, (2, 3), floating=floating)


def _forward(model: AgentNetwork, observations: Array) -> Array:
    return model(observations)[0]


def _loss(model: AgentNetwork, observations: Array) -> Array:
    return jnp.mean(jnp.square(_forward(model, observations)))


def _assert_compile(case: NetworkCase, leading_shape: tuple[int, ...], *, floating: bool) -> None:
    model = make_network(case)
    shape = (*leading_shape, case.config.input_features)
    dtype = jnp.float32 if floating else jnp.int32
    spec = jax.ShapeDtypeStruct(shape, dtype)
    compiled = nnx.jit(_forward).trace(model, spec).lower().compile()
    observations = (jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape) % 5 - 1).astype(dtype)
    output = compiled(model, observations)
    assert output.shape == (*leading_shape, case.config.output_features)
    assert output.dtype == jnp.float32
    np.testing.assert_allclose(output, _forward(model, observations), rtol=2e-5, atol=2e-6)
    if case.kind == "actor":
        np.testing.assert_allclose(output.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)
        assert jnp.all(output >= 0)
    gradient = nnx.jit(nnx.value_and_grad(_loss)).trace(model, spec).lower().compile()
    loss, state = gradient(model, observations)
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(state))
