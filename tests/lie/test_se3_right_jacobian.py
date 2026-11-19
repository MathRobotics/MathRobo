import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

from mathrobo import SE3


def augmented(xi):
    matrix = np.zeros((12, 12))
    matrix[:6, :6] = -SE3.hat_adj(np.asarray(xi, dtype=np.float64))
    return matrix


def reference(xi, u, v):
    matrix = augmented(xi)
    matrix[:6, 6:] = np.eye(6)
    direction_u, direction_v = augmented(u), augmented(v)

    def ordered_second(first, second):
        zero = np.zeros_like(matrix)
        block = np.block([[matrix, first, zero],
                          [zero, matrix, second],
                          [zero, zero, matrix]])
        return expm(block)[:6, 30:]

    return (expm(matrix)[:6, 6:],
            expm_frechet(matrix, direction_u, compute_expm=False)[:6, 6:],
            ordered_second(direction_u, direction_v) + ordered_second(direction_v, direction_u))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_right_jacobian_against_block_exponentials(dtype):
    axis = np.array([2., -3., 6.]) / 7
    w = np.array([0, 0, 1e-10, .00101, .35, 1.99999, 2.00001, np.pi - 1e-5, 7.])
    rho = np.broadcast_to([.3, -.2, .4], (len(w), 3)).copy()
    rho[0] = 0
    xi = np.concatenate((w[:, None] * axis, rho), axis=-1).astype(dtype)
    u = np.array([.6, -.3, .4, -.25, .15, .75], dtype=dtype)
    v = np.array([-.2, .5, .4, .7, -.15, -.25], dtype=dtype)
    with np.errstate(divide='raise', invalid='raise'):
        matrices = [SE3.right_jacobian(xi), SE3.right_jacobian_directional(xi, u),
                    SE3.right_jacobian_second_directional(xi, u, v)]
    tolerance = 2e-6 if dtype == np.float32 else 6e-13
    for actual, expected in zip(matrices, zip(*(reference(x, u, v) for x in xi))):
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(matrices[2], SE3.right_jacobian_second_directional(xi, v, u),
                               atol=tolerance, rtol=tolerance)


def test_right_jacobian_body_velocity_and_central_difference():
    xi = np.array([.7, -.3, .4, -.2, .3, .1])
    u = np.array([-.4, .2, .3, .1, .5, -.4])
    v = np.array([.3, -.2, .7, .2, -.1, -.5])
    h = 1e-5
    transform = SE3.set_mat(SE3.exp(xi))
    relative_derivative = transform.mat_inv() @ (SE3.exp(xi + h * u) - SE3.exp(xi - h * u)) / (2 * h)
    np.testing.assert_allclose(SE3.vee(relative_derivative), SE3.right_jacobian(xi) @ u, atol=2e-10)
    difference = (SE3.right_jacobian_directional(xi + h * v, u)
                  - SE3.right_jacobian_directional(xi - h * v, u)) / (2 * h)
    np.testing.assert_allclose(difference, SE3.right_jacobian_second_directional(xi, u, v), atol=2e-10)


def test_jit_directional_jacobian_with_broadcast_and_jvp():
    xi = jnp.array([[[0., 0., 0., .3, -.2, .4]], [[2.2, -.4, .1, -.2, .3, .1]]], dtype=jnp.float32)
    u = jnp.array([[.5, -.3, .2, -.25, .15, .75], [.2, -.1, .3, .1, -.2, .4]], dtype=xi.dtype)
    v = -.3 * jnp.ones(6, dtype=xi.dtype)
    first = jax.jit(lambda x, d: SE3.right_jacobian_directional(x, d, LIB='jax'))
    second = jax.jit(lambda x, d, e: SE3.right_jacobian_second_directional(x, d, e, LIB='jax'))
    value = jax.jit(lambda x: SE3.right_jacobian(x, LIB='jax'))
    np.testing.assert_allclose(value(xi), SE3.right_jacobian(np.asarray(xi)), atol=2e-6)
    np.testing.assert_allclose(first(xi, u), SE3.right_jacobian_directional(np.asarray(xi), np.asarray(u)), atol=2e-6)
    actual = second(xi, u, v)
    assert actual.shape == (2, 2, 6, 6)
    assert actual.dtype == xi.dtype
    expected = jax.jvp(lambda x: first(x, u), (xi,), (jnp.broadcast_to(v, xi.shape),))[1]
    np.testing.assert_allclose(actual, expected, atol=2e-6)
