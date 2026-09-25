"""Small-angle values and derivatives against independent matrix exponentials."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

from mathrobo import SO3, SE3


# Support both the minimum JAX version and the current public context manager.
enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64


ANGLES = np.array([0, 1e-300, 1e-100, 1e-34, 1e-16, 1e-8, 1e-4,
                   0.249999, 0.25, 0.250001, 0.7, 2.0])
METHODS = [(SO3.exp_integ, SO3.hat, 1),
           (SO3.exp_integ2nd, SO3.hat, 2),
           (SE3.exp_integ_adj, SE3.hat_adj, 1)]


def inputs(index, dtype):
    direction = np.array([2., -3., 6.]) / 7
    vec = ANGLES[:, None] * direction
    if index == 2:
        vec = np.concatenate((vec, np.broadcast_to([0.7, -1.2, 0.4], vec.shape)), axis=-1)
    # Tiny inputs intentionally underflow when cast to float32.
    with np.errstate(under='ignore'):
        return vec.astype(dtype)


def augmented(vec, index):
    _, hat, order = METHODS[index]
    h = hat(vec.astype(np.float64))
    n = len(h)
    m = np.zeros(((order+1)*n, (order+1)*n))
    m[:n, :n] = h
    for k in range(order):
        m[k*n:(k+1)*n, (k+1)*n:(k+2)*n] = np.eye(n)
    return m, n


def reference(vec, a, index):
    m, n = augmented(vec, index)
    return expm(a*m)[:n, -n:]


@lru_cache(None)
def evaluator(index, mode, derivatives=False):
    method = METHODS[index][0]
    if mode == 'numpy':
        return lambda v, a: method(v, a, 'numpy')
    fun = lambda v, a: method(v, a, 'jax')
    if derivatives:
        fun = jax.vmap(jax.jacrev(fun, argnums=(0, 1)), in_axes=(0, None))
    return jax.jit(fun) if mode == 'jit' else fun


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('mode', ['numpy', 'eager', 'jit'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('a', [0., 0.7, -2., 2.])
def test_integral_values(index, mode, dtype, a):
    with enable_x64(True):
        vec = inputs(index, dtype)
        expected = np.stack([reference(v, a, index) for v in vec])
        evaluate = evaluator(index, mode)
        arg = vec if mode == 'numpy' else jnp.asarray(vec)
        # Raise for divisions and invalid intermediates even in unselected branches.
        with np.errstate(divide='raise', invalid='raise', over='raise'):
            batch = np.asarray(evaluate(arg.reshape((2, 6, -1)), a)).reshape(expected.shape)
            single = np.stack([np.asarray(evaluate(v, a)) for v in arg])
        assert np.isfinite(batch).all()
        assert np.isfinite(single).all()
        tol = 3e-5 if dtype == np.float32 else 3e-12
        np.testing.assert_allclose(batch, expected, rtol=tol, atol=tol)
        np.testing.assert_allclose(single, batch, rtol=tol, atol=tol)
        if mode != 'numpy':
            mapped = jax.vmap(lambda v: METHODS[index][0](v, a, 'jax'))
            if mode == 'jit':
                mapped = jax.jit(mapped)
            np.testing.assert_allclose(mapped(arg), batch, rtol=tol, atol=tol)


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('mode', ['eager', 'jit'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('a', [0., -2., 2.])
def test_integral_gradients(index, mode, dtype, a):
    with enable_x64(True):
        vec = inputs(index, dtype)
        dv, da = evaluator(index, mode, derivatives=True)(jnp.asarray(vec), a)
        assert np.isfinite(dv).all()
        assert np.isfinite(da).all()
        expected_dv, expected_da = [], []
        for v in vec:
            tiny = np.linalg.norm(a * v[:3].astype(np.float64)) < 1e-12
            effective = v.copy()
            if tiny:
                effective[:3] = 0
            m, n = augmented(effective, index)
            jac = []
            for basis in np.eye(len(v)):
                direction = np.zeros_like(m)
                if not tiny or np.any(basis[3:]):
                    direction[:n, :n] = METHODS[index][1](basis)
                jac.append(expm_frechet(a*m, a*direction, compute_expm=False)[:n, -n:])
            expected_dv.append(np.stack(jac, axis=-1))
            expected_da.append((m @ expm(a*m))[:n, -n:])
        tol = 2e-4 if dtype == np.float32 else 2e-11
        np.testing.assert_allclose(dv, expected_dv, rtol=tol, atol=tol)
        np.testing.assert_allclose(da, expected_da, rtol=tol, atol=tol)


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('mode', ['eager', 'jit'])
def test_native_batch_gradients(index, mode):
    with enable_x64(True):
        vec = jnp.asarray(inputs(index, np.float64).reshape(2, 6, -1))
        method = METHODS[index][0]
        weights = jnp.arange((6 if index == 2 else 3)**2).reshape(
            (6, 6) if index == 2 else (3, 3)) / 10
        fun = jax.grad(lambda v, a: jnp.sum(method(v, a, 'jax') * weights), argnums=(0, 1))
        if mode == 'jit':
            fun = jax.jit(fun)
        dv, da = fun(vec, 2.)
        single_dv, single_da = evaluator(index, mode, derivatives=True)(vec.reshape(12, -1), 2.)
        assert np.isfinite(dv).all()
        assert np.isfinite(da).all()
        np.testing.assert_allclose(dv.reshape(12, -1),
                                   np.einsum('bijk,ij->bk', single_dv, weights), atol=2e-12)
        np.testing.assert_allclose(da, np.einsum('bij,ij->', single_da, weights), atol=2e-12)


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('mode', ['numpy', 'eager', 'jit'])
@pytest.mark.parametrize('a', [-2., 0.7, 2.])
def test_zero_rotation_limit(index, mode, a):
    with enable_x64(True):
        # Effective angles below, at, and above the strict 1e-12 cutoff.
        angles = np.array([0., 1e-300, 1e-34, 1e-16, 0.999e-12, 1e-12, 1.001e-12])
        vec = np.zeros((len(angles), 6 if index == 2 else 3))
        vec[:, 0] = angles / abs(a)
        if index == 2:
            vec[:, 3:] = [0.7, -1.2, 0.4]
        arg = vec if mode == 'numpy' else jnp.asarray(vec)
        result = np.asarray(evaluator(index, mode)(arg, a))
        n = result.shape[-1]
        limit = (a*a/2 if index == 1 else a) * np.eye(n)
        if index == 2:
            limit[3:, :3] = (a*a/2) * SO3.hat(vec[0, 3:])
        # No residual rotation corrections may survive in the tiny region.
        np.testing.assert_array_equal(result[:5], np.broadcast_to(result[0], result[:5].shape))
        # Translation products may round differently depending on multiplication order.
        np.testing.assert_allclose(result[0], limit, rtol=3e-16 if index == 2 else 0, atol=0)
        for v, actual in zip(vec[5:], result[5:]):
            np.testing.assert_allclose(actual, reference(v, a, index), rtol=2e-15, atol=1e-27)
            assert actual[1, 2] != 0
