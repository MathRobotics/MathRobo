import jax
import jax.numpy as jnp
import numpy as np

from scipy import integrate
from scipy.linalg import expm

import mathrobo as mr


def _so3_exp_integral_reference(vec: np.ndarray, a: float) -> np.ndarray:
    def integrand(s: float) -> np.ndarray:
        return expm(s * mr.SO3.hat(vec, "numpy"))

    mat, _ = integrate.quad_vec(integrand, 0.0, a)
    return mat


def _se3_exp_integral_reference(vec: np.ndarray, a: float) -> np.ndarray:
    def integrand(s: float) -> np.ndarray:
        return expm(s * mr.SE3.hat_adj(vec, "numpy"))

    mat, _ = integrate.quad_vec(integrand, 0.0, a)
    return mat


def test_so3_exp_jax_matches_numpy():
    vec = np.array([0.3, -0.6, 0.9], dtype=np.float64)
    a = 0.75

    expected = mr.SO3.exp(vec, a, "numpy")
    actual = np.array(mr.SO3.exp(jnp.array(vec), a, "jax"))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_so3_exp_integ_jax_matches_integral_reference():
    vec = np.array([0.3, -0.6, 0.9], dtype=np.float64)
    a = 0.75

    expected = _so3_exp_integral_reference(vec, a)
    actual = np.array(mr.SO3.exp_integ(jnp.array(vec), a, "jax"))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_se3_exp_jax_matches_matrix_exponential():
    vec = np.array([0.3, -0.6, 0.9, 1.2, -0.4, 0.7], dtype=np.float64)
    a = 0.5

    expected = expm(a * mr.SE3.hat(vec, "numpy"))
    actual = np.array(mr.SE3.exp(jnp.array(vec), a, "jax"))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_se3_exp_integ_adj_jax_matches_integral_reference():
    vec = np.array([0.3, -0.6, 0.9, 1.2, -0.4, 0.7], dtype=np.float64)
    a = 0.5

    expected = _se3_exp_integral_reference(vec, a)
    actual = np.array(mr.SE3.exp_integ_adj(jnp.array(vec), a, "jax"))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_se3wrench_mat_adj_jax_matches_numpy():
    rot = np.array(
        [
            [0.36, -0.48, 0.8],
            [0.8, 0.60, 0.0],
            [-0.48, 0.64, 0.60],
        ],
        dtype=np.float64,
    )
    pos = np.array([1.0, -2.0, 3.0], dtype=np.float64)

    expected = mr.SE3wrench(rot, pos, "numpy").mat_adj()
    actual = np.array(mr.SE3wrench(jnp.array(rot), jnp.array(pos), "jax").mat_adj())

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_se3wrench_mat_inv_adj_jax_is_jittable():
    def func(p: jnp.ndarray) -> jnp.ndarray:
        return mr.SE3wrench(jnp.eye(3), p, "jax").mat_inv_adj()

    out = jax.jit(func)(jnp.array([1.0, 2.0, 3.0]))

    assert isinstance(out, jax.Array)
    assert out.shape == (6, 6)
