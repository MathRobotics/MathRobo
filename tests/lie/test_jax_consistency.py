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


def test_se3_exp_adj_jax_matches_numpy():
    vec = np.array([0.3, -0.6, 0.9, 1.2, -0.4, 0.7], dtype=np.float64)
    a = 0.5

    expected = mr.SE3.exp_adj(vec, a, "numpy")
    actual = np.array(mr.SE3.exp_adj(jnp.array(vec), a, "jax"))

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


def test_jax_quaternion_constructors_return_jax_matrices():
    quaternion = jnp.array([1.0, 0.0, 0.0, 0.0])

    so3_mat = mr.SO3.set_quaternion(quaternion, "jax").mat()
    se3_mat = mr.SE3.set_pos_quaternion(jnp.zeros(3), quaternion, "jax").mat()

    assert isinstance(so3_mat, jax.Array)
    assert isinstance(se3_mat, jax.Array)
    assert so3_mat.shape == (3, 3)
    assert se3_mat.shape == (4, 4)


def test_jax_matmul_vectors_return_arrays():
    so3_vec = mr.SO3.eye("jax") @ jnp.ones(3)
    se3_pos = mr.SE3.eye("jax") @ jnp.ones(3)
    se3_adj = mr.SE3.eye("jax") @ jnp.ones(6)

    assert isinstance(so3_vec, jax.Array)
    assert isinstance(se3_pos, jax.Array)
    assert isinstance(se3_adj, jax.Array)
    assert so3_vec.shape == (3,)
    assert se3_pos.shape == (3,)
    assert se3_adj.shape == (6,)


def test_se3_commute_helpers_jax_match_numpy():
    vec = np.arange(1.0, 7.0)

    np.testing.assert_allclose(
        np.array(mr.SE3.hat_commute(jnp.array(vec), "jax")),
        mr.SE3.hat_commute(vec, "numpy"),
    )
    np.testing.assert_allclose(
        np.array(mr.SE3wrench.hat_commute(jnp.array(vec), "jax")),
        mr.SE3wrench.hat_commute(vec, "numpy"),
    )
    np.testing.assert_allclose(
        np.array(mr.SE3wrench.hat_commute_adj(jnp.array(vec), "jax")),
        mr.SE3wrench.hat_commute_adj(vec, "numpy"),
    )


def test_inertia_jax_matches_numpy():
    so3_vec = np.arange(1.0, 7.0)
    se3_hat_vec = np.arange(1.0, 11.0)
    se3_commute_vec = np.arange(1.0, 7.0)

    np.testing.assert_allclose(
        np.array(mr.SO3inertia.hat(jnp.array(so3_vec), "jax")),
        mr.SO3inertia.hat(so3_vec, "numpy"),
    )
    np.testing.assert_allclose(
        np.array(mr.SO3inertia.hat_commute(jnp.array(so3_vec[:3]), "jax")),
        mr.SO3inertia.hat_commute(so3_vec[:3], "numpy"),
    )
    np.testing.assert_allclose(
        np.array(mr.SE3inertia.hat(jnp.array(se3_hat_vec), "jax")),
        mr.SE3inertia.hat(se3_hat_vec, "numpy"),
    )
    np.testing.assert_allclose(
        np.array(mr.SE3inertia.hat_commute(jnp.array(se3_commute_vec), "jax")),
        mr.SE3inertia.hat_commute(se3_commute_vec, "numpy"),
    )
