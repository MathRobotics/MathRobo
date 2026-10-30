import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mathrobo import CMVector


@pytest.mark.parametrize("order", [0, 1, 5, 11, 14])
@pytest.mark.parametrize("batch", [(), (2, 3)])
def test_derivative_and_coefficient_conventions(order, batch):
    vecs = np.random.default_rng(0).normal(size=batch + (order, 6))
    factors = np.asarray([math.factorial(k) for k in range(order)])[:, None]
    expected_cm = vecs / factors
    normal = CMVector(vecs)
    direct = CMVector.set_cmvecs(expected_cm)

    for obj in (normal, direct):
        np.testing.assert_allclose(obj.cm_vecs(), expected_cm)
        np.testing.assert_allclose(obj.vecs(), vecs)
        np.testing.assert_allclose(obj.cm_vec(), expected_cm.reshape(batch + (order * 6,)))
        np.testing.assert_allclose(obj.vec(), vecs.reshape(batch + (order * 6,)))
    # Direct construction must retain coefficients without a lossy round trip.
    assert direct.cm_vecs() is expected_cm


def test_one_dimensional_series_layouts():
    vecs = np.arange(1.0, 6.0)
    factors = np.asarray([math.factorial(k) for k in range(len(vecs))])
    normal = CMVector(vecs)
    np.testing.assert_allclose(normal.vecs(), vecs)
    np.testing.assert_allclose(normal.cm_vecs(), vecs / factors)
    assert normal.vec().shape == normal.cm_vec().shape == (5,)

    direct = CMVector.set_cmvecs(vecs / factors)
    np.testing.assert_allclose(direct.vecs(), vecs[:, None])
    assert direct.cm_vecs().shape == (5, 1)
    assert direct.vec().shape == direct.cm_vec().shape == (5,)


@pytest.mark.parametrize("order", [5, 11, 14])
def test_jax_jit_and_gradients(order):
    x = jnp.arange(order * 6, dtype=jnp.float32).reshape(order, 6) / 10
    factors = np.asarray([math.factorial(k) for k in range(order)], dtype=np.float32)[:, None]

    @jax.jit
    def normalized(values):
        return CMVector(values).cm_vecs()

    @jax.jit
    def ordinary(values):
        return CMVector.set_cmvecs(values).vecs()

    np.testing.assert_allclose(normalized(x), np.asarray(x) / factors, rtol=2e-6)
    np.testing.assert_allclose(ordinary(normalized(x)), x, rtol=2e-6)
    gradient = jax.jit(jax.grad(lambda values: ordinary(values).sum()))(x)
    np.testing.assert_allclose(gradient, np.broadcast_to(factors, x.shape), rtol=2e-6)
    assert normalized(x).dtype == x.dtype
