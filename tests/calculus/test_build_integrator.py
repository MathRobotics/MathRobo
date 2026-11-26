import numpy as np
import pytest

import mathrobo as mr

def test_output_shapes():
        dof, order, dt = 3, 2, 0.01
        for method in ["euler", "poly", "rk2", "rk4"]:
                A, B = mr.build_integrator(dof, order, dt, method)
                assert A.shape == (dof * order, dof * order)
                assert B.shape == (dof * order, dof)

def test_euler_values():
        dof, order, dt = 1, 2, 0.1
        A, B = mr.build_integrator(dof, order, dt, method="euler")
        expected_A = np.eye(2)
        expected_A[0, 1] = dt
        expected_B = np.zeros((2, 1))
        expected_B[1, 0] = dt
        np.testing.assert_allclose(A, expected_A, rtol=1e-10)
        np.testing.assert_allclose(B, expected_B, rtol=1e-10)

def test_poly_values():
        dof, order, dt = 1, 3, 0.1
        A, B = mr.build_integrator(dof, order, dt, method="poly")
        expected_A = np.eye(3)
        expected_A[0, 1] = dt
        expected_A[0, 2] = (dt**2) / 2
        expected_A[1, 2] = dt
        expected_B = np.zeros((3, 1))
        expected_B[0, 0] = (dt**3) / 6
        expected_B[1, 0] = (dt**2) / 2
        expected_B[2, 0] = dt
        np.testing.assert_allclose(A, expected_A, rtol=1e-10)
        np.testing.assert_allclose(B, expected_B, rtol=1e-10)

def test_rk2_values():
        dof, order, dt = 1, 2, 0.1
        A, B = mr.build_integrator(dof, order, dt, method="rk2")
        np.testing.assert_allclose(A, [[1.0, dt], [0.0, 1.0]], rtol=1e-12)
        np.testing.assert_allclose(B, [[dt**2 / 2], [dt]], rtol=1e-12)


def _rk_step(x, u, dt, dof, order, method):
        def f(state):
                return np.concatenate([state[dof:], u])
        if method == "euler":
                return x + dt * f(x)
        if method == "rk2":
                return x + dt * f(x + 0.5 * dt * f(x))
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


@pytest.mark.parametrize("method", ["euler", "rk2", "rk4"])
@pytest.mark.parametrize("order", [1, 2, 3, 5])
def test_matches_explicit_runge_kutta_step(method, order):
        dof, dt = 2, 0.3
        rng = np.random.default_rng(order)
        x = rng.normal(size=dof * order)
        u = rng.normal(size=dof)
        A, B = mr.build_integrator(dof, order, dt, method=method)
        np.testing.assert_allclose(A @ x + B @ u, _rk_step(x, u, dt, dof, order, method), rtol=1e-12, atol=1e-12)


def test_rk4_approx():
        dof, order, dt = 1, 3, 0.1
        A, B = mr.build_integrator(dof, order, dt, method="rk4")
        expected_A = np.eye(3)
        # Coefficient-based: coefficients for 1st, 2nd, and 3rd order terms (1, 0.5, 1/6)
        expected_A[0, 1] = 1 * dt
        expected_A[0, 2] = 0.5 * dt**2
        expected_A[1, 2] = 1 * dt
        expected_B = np.zeros((3, 1))
        expected_B[0, 0] = (1/6) * dt**3
        expected_B[1, 0] = 0.5 * dt**2
        expected_B[2, 0] = 1 * dt
        np.testing.assert_allclose(A, expected_A, rtol=1e-10)
        np.testing.assert_allclose(B, expected_B, rtol=1e-10)

def test_invalid_method():
        with pytest.raises(ValueError, match="Unsupported method"):
                mr.build_integrator(1, 2, 0.1, method="invalid")


def test_poly_is_exact_discretization_for_high_order():
        from scipy.linalg import expm
        dof, order, dt = 2, 7, 0.4
        dim = dof * order
        S = np.zeros((dim + dof, dim + dof))
        S[:dim, dof:] = np.eye(dim)
        exact = expm(S * dt)
        A, B = mr.build_integrator(dof, order, dt, method="poly")
        np.testing.assert_allclose(A, exact[:dim, :dim], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(B, exact[:dim, dim:], rtol=1e-12, atol=1e-15)
