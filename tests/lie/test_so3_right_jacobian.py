import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mathrobo as mr

AXIS = np.array([2., -3., 6.]) / 7
ANGLES = np.array([0, 1e-10, .00101, .35, 1.99999, 2.00001, np.pi - 1e-5, 7.])


def embed(vec):
    """SE3 exponential coordinates (phi, 0) of a rotation vector."""
    vec = np.asarray(vec)
    return np.concatenate((vec, np.zeros_like(vec)), axis=-1)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_right_jacobians_match_se3_rotation_block(dtype):
    phi = (ANGLES[:, None] * AXIS).astype(dtype)
    u = np.array([.6, -.3, .4], dtype=dtype)
    v = np.array([-.2, .5, .4], dtype=dtype)
    with np.errstate(divide='raise', invalid='raise'):
        actual = [mr.SO3.right_jacobian(phi), mr.SO3.right_jacobian_directional(phi, u),
                  mr.SO3.right_jacobian_second_directional(phi, u, v)]
    expected = [mr.SE3.right_jacobian(embed(phi)), mr.SE3.right_jacobian_directional(embed(phi), embed(u)),
                mr.SE3.right_jacobian_second_directional(embed(phi), embed(u), embed(v))]
    tolerance = 1e-6 if dtype == np.float32 else 1e-14
    for a, e in zip(actual, expected):
        assert a.shape == phi.shape + (3,) and a.dtype == dtype
        np.testing.assert_allclose(a, e[..., :3, :3], atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('cls', [mr.SO3, mr.SO3wrench])
def test_right_jacobian_definition(cls):
    phi, u, h = np.array([.7, -.3, .4]), np.array([-.4, .2, .3]), 1e-5
    exp = lambda x: cls.exp(x, 1.)
    derivative = exp(phi).T @ (exp(phi + h * u) - exp(phi - h * u)) / (2 * h)
    np.testing.assert_allclose(cls.vee(derivative), cls.right_jacobian(phi) @ u, atol=1e-10)

    v = np.array([.3, -.2, .7])
    np.testing.assert_allclose(
        cls.right_jacobian_directional(phi, u),
        (cls.right_jacobian(phi + h * u) - cls.right_jacobian(phi - h * u)) / (2 * h), atol=1e-9)
    np.testing.assert_allclose(
        cls.right_jacobian_second_directional(phi, u, v),
        (cls.right_jacobian_directional(phi + h * v, u) - cls.right_jacobian_directional(phi - h * v, u)) / (2 * h),
        atol=1e-9)


def _series(order, batch=(), seed=0):
    return np.random.default_rng(seed).normal(size=batch + (order, 3)) * 0.8


@pytest.mark.parametrize('order', [1, 2, 4])
def test_coefficients_match_se3_rotation_block(order):
    coordinates = _series(order, (5,))
    direction = _series(order, (5,), seed=1)
    np.testing.assert_allclose(mr.CMTM.so3_right_jacobian_coefficients(coordinates),
                               mr.CMTM.se3_right_jacobian_coefficients(embed(coordinates))[..., :3, :3],
                               atol=1e-13)
    np.testing.assert_allclose(mr.CMTM.so3_right_jacobian_coefficients_jvp(coordinates, direction),
                               mr.CMTM.se3_right_jacobian_coefficients_jvp(
                                   embed(coordinates), embed(direction))[..., :3, :3],
                               atol=1e-13)


@pytest.mark.parametrize('order', [1, 3])
def test_coefficient_vjp_is_transpose_of_jvp(order):
    coordinates, direction = _series(order, (4,)), _series(order, (4,), seed=1)
    cotangent = np.random.default_rng(2).normal(size=coordinates.shape + (3,))
    jvp = mr.CMTM.so3_right_jacobian_coefficients_jvp(coordinates, direction)
    vjp = mr.CMTM.so3_right_jacobian_coefficients_vjp(coordinates, cotangent)
    assert vjp.shape == coordinates.shape
    np.testing.assert_allclose(np.sum(jvp * cotangent), np.sum(vjp * direction), rtol=1e-12)


@pytest.mark.parametrize('order', [1, 2, 4])
def test_motion_matches_se3_with_zero_translation(order):
    coordinates = _series(order, (3,))
    motion, jacobian = mr.CMTM.from_so3_exponential_coordinates(coordinates, return_jacobian=True)
    reference, reference_jacobian = mr.CMTM.from_se3_exponential_coordinates(
        embed(coordinates), return_jacobian=True)

    assert type(motion._mat) is mr.SO3
    np.testing.assert_allclose(motion.elem_mat(), reference._mat.rot(), atol=1e-13)
    np.testing.assert_allclose(motion.cmvecs().cm_vecs(), reference.cmvecs().cm_vecs()[..., :3], atol=1e-13)
    np.testing.assert_allclose(reference.cmvecs().cm_vecs()[..., 3:], 0, atol=1e-13)
    np.testing.assert_allclose(jacobian, reference_jacobian[..., :3, :3], atol=1e-13)
    without = mr.CMTM.from_so3_exponential_coordinates(mr.CMVector.set_cmvecs(coordinates))
    np.testing.assert_allclose(without.mat(), motion.mat())


def test_jax_jit_matches_numpy():
    coordinates = _series(3, (2,))

    @jax.jit
    def run(c):
        motion, jacobian = mr.CMTM.from_so3_exponential_coordinates(c, 'jax', return_jacobian=True)
        return motion.mat(), jacobian, mr.SO3.right_jacobian(c[..., 0, :], 'jax')

    mat, jacobian, jr = run(jnp.asarray(coordinates, dtype=jnp.float32))
    reference, reference_jacobian = mr.CMTM.from_so3_exponential_coordinates(coordinates, return_jacobian=True)
    np.testing.assert_allclose(mat, reference.mat(), atol=1e-5)
    np.testing.assert_allclose(jacobian, reference_jacobian, atol=1e-5)
    np.testing.assert_allclose(jr, mr.SO3.right_jacobian(coordinates[..., 0, :]), atol=1e-5)


@pytest.mark.parametrize('bad', [np.zeros((2, 6)), np.zeros(3), np.zeros((0, 3))])
def test_coordinate_shapes_are_validated(bad):
    with pytest.raises(ValueError, match='SO3 coordinate'):
        mr.CMTM.so3_right_jacobian_coefficients(bad)
    with pytest.raises(ValueError, match='shape'):
        mr.SO3.right_jacobian(np.zeros(6))
