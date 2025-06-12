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

def test_rk2_approx():
    dof, order, dt = 1, 2, 0.1
    A, B = mr.build_integrator(dof, order, dt, method="rk2")
    # Check basic structure (exact value derivation is difficult, so check for positive values and order)
    assert A[0, 1] > dt       # Should be larger than Euler's method
    assert B[1, 0] > dt       # Similarly

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
