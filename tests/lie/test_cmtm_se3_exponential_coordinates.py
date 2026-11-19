import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from mathrobo import CMTM, CMVector, SE3


def lifted_generator(coefficients):
    order, size, _ = coefficients.shape
    result = np.zeros((order * size, order * size))
    for row in range(order):
        for column in range(row + 1):
            result[row * size:(row + 1) * size, column * size:(column + 1) * size] = coefficients[row - column]
    return result


def jacobian_reference(coordinates):
    generator = lifted_generator(SE3.hat_adj(-coordinates.astype(np.float64)))
    size = generator.shape[0]
    augmented = np.zeros((2 * size, 2 * size))
    augmented[:size, :size] = generator
    augmented[:size, size:] = np.eye(size)
    return expm(augmented)[:size, size:size + 6].reshape(-1, 6, 6)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_series_against_independent_lifted_exponentials(dtype):
    rng = np.random.default_rng(9)
    coordinates = rng.normal(scale=.3, size=(2, 3, 7, 6)).astype(dtype)
    angles = np.array([[0., 1e-9, 1.99999], [2.00001, np.pi, 7.]])
    coordinates[..., 0, :3] = angles[..., None] * np.array([2., -3., 6.]) / 7
    motion, jacobian = CMTM.from_se3_exponential_coordinates(coordinates, return_jacobian=True)
    velocity_only = CMTM.from_se3_exponential_coordinates(coordinates)
    expected_jacobian = np.stack([jacobian_reference(x) for x in coordinates.reshape(-1, 7, 6)])
    expected_motion = np.stack([expm(lifted_generator(SE3.hat(x.astype(np.float64))))
                               for x in coordinates.reshape(-1, 7, 6)])
    tolerance = 3e-6 if dtype == np.float32 else 4e-12
    assert motion.cmvecs().cm_vecs().shape == (2, 3, 6, 6)
    assert jacobian.dtype == motion.cmvecs().cm_vecs().dtype == dtype
    np.testing.assert_allclose(jacobian.reshape(-1, 7, 6, 6), expected_jacobian, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(motion.mat().reshape(-1, 28, 28), expected_motion, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(velocity_only.cmvecs().cm_vecs(), motion.cmvecs().cm_vecs(),
                               atol=tolerance, rtol=tolerance)


def test_factorial_convention_and_directional_coefficients():
    ordinary = np.array([[.2, -.3, .1, -.2, .6, .2],
                         [.2, .1, .4, -.3, .2, .4],
                         [.4, -.1, -.2, .1, .9, -.3]])
    x, u, v = ordinary
    coordinates = CMVector(ordinary)
    motion, jacobian = CMTM.from_se3_exponential_coordinates(coordinates, return_jacobian=True)
    jr = SE3.right_jacobian(x)
    directional = SE3.right_jacobian_directional(x, u)
    second = (SE3.right_jacobian_second_directional(x, u, u)
              + SE3.right_jacobian_directional(x, v)) / 2
    np.testing.assert_allclose(jacobian, [jr, directional, second], atol=1e-14)
    np.testing.assert_allclose(motion.vecs(), [jr @ u, jr @ v + directional @ u], atol=1e-14)
    np.testing.assert_allclose(CMTM.se3_right_jacobian_coefficients(coordinates), jacobian, atol=1e-14)


def test_constant_and_invalid_series():
    coordinates = np.array([[0, 0, 0, 2, 3, 4]])
    motion, jacobian = CMTM.from_se3_exponential_coordinates(coordinates, return_jacobian=True)
    np.testing.assert_array_equal(motion.elem_mat()[:3, 3], coordinates[0, 3:])
    assert motion.cmvecs().cm_vecs().shape == (0, 6)
    np.testing.assert_array_equal(jacobian[0], SE3.right_jacobian(coordinates[0]))
    for invalid in (np.zeros(6), np.zeros((0, 6)), np.zeros((3, 5))):
        with pytest.raises(ValueError, match='coordinate coefficients'):
            CMTM.from_se3_exponential_coordinates(invalid)
        with pytest.raises(ValueError, match='coordinate coefficients'):
            CMTM.se3_right_jacobian_coefficients(invalid)


def test_force_projection_includes_time_derivative_of_jacobian():
    x = np.array([.2, -.3, .1, -.2, .6, .2])
    u = np.array([.2, .1, .4, -.3, .2, .4])
    force = np.array([.7, -.3, .5, .2, -.8, .4])
    force_derivative = np.array([.4, .9, .2, -.1, .7, -.6])
    jacobian, rate = CMTM.se3_right_jacobian_coefficients(np.stack((x, u)))
    expected = rate.T @ force + jacobian.T @ force_derivative
    h = 1e-5
    plus = SE3.right_jacobian(x + h * u).T @ (force + h * force_derivative)
    minus = SE3.right_jacobian(x - h * u).T @ (force - h * force_derivative)
    np.testing.assert_allclose(expected, (plus - minus) / (2 * h), atol=2e-10)


def test_jit_batched_coefficients_and_zero_rotation_gradient():
    coordinates = jnp.asarray(np.random.default_rng(3).normal(scale=.2, size=(2, 5, 6)), dtype=jnp.float32)
    coordinates = coordinates.at[:, 0, :3].set(jnp.array([[0., 0., 0.], [2.2, -.4, .1]]))

    @jax.jit
    def compute(x):
        motion, jacobian = CMTM.from_se3_exponential_coordinates(x, LIB='jax', return_jacobian=True)
        return motion.mat(), motion.cmvecs().cm_vecs(), jacobian

    matrix, velocity, jacobian = compute(coordinates)
    assert matrix.dtype == velocity.dtype == jacobian.dtype == coordinates.dtype
    reference, jr = CMTM.from_se3_exponential_coordinates(np.asarray(coordinates), return_jacobian=True)
    np.testing.assert_allclose(matrix, reference.mat(), atol=2e-6)
    np.testing.assert_allclose(velocity, reference.cmvecs().cm_vecs(), atol=2e-6)
    np.testing.assert_allclose(jacobian, jr, atol=2e-6)
    gradient = jax.grad(lambda x: compute(x)[0].sum() + compute(x)[2].sum())(coordinates[:1])
    assert np.isfinite(gradient).all()
