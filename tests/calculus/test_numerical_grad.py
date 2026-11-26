import pytest

import numpy as np
import mathrobo as mr

def test_scalar_function():
        f = lambda x: np.sum(x**2)
        x = np.array([1.0, 2.0, -3.0])
        expected = 2 * x
        grad = mr.numerical_grad(x, f, method="central")
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

def test_vector_function():
        f = lambda x: np.array([x[0]**2, np.sin(x[1]), np.exp(x[2])])
        x = np.array([2.0, np.pi/4, 0.0])
        expected = np.array([
                [2*x[0], 0, 0],
                [0, np.cos(x[1]), 0],
                [0, 0, np.exp(x[2])]
        ])
        grad = mr.numerical_grad(x, f, method="central")
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

def test_forward_vs_central():
        f = lambda x: np.sum(np.sin(x))
        x = np.array([0.5, 1.0, 1.5])
        grad_f = mr.numerical_grad(x, f, method="forward")
        grad_c = mr.numerical_grad(x, f, method="central")
        # Central should be more accurate
        exact = np.cos(x)
        assert np.linalg.norm(grad_c - exact) < np.linalg.norm(grad_f - exact)

def test_fourth_order_accuracy():
        f = lambda x: np.sum(np.sin(x))
        x = np.array([1.0, 2.0])
        grad_central = mr.numerical_grad(x, f, method="central")
        grad_fourth = mr.numerical_grad(x, f, method="fourth")
        exact = np.cos(x)
        assert np.linalg.norm(grad_fourth - exact) < np.linalg.norm(grad_central - exact)

def test_invalid_method_raises():
        f = lambda x: np.sum(x)
        x = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
                mr.numerical_grad(x, f, method="invalid")


def test_default_steps_are_accurate_for_each_method():
        f = lambda x: np.sum(np.sin(x))
        x = np.array([0.3, 1.1, -0.7])
        tolerances = {"forward": 1e-7, "central": 1e-10, "fourth": 1e-12}
        for method, tol in tolerances.items():
                np.testing.assert_allclose(mr.numerical_grad(x, f, method=method), np.cos(x), atol=tol)


def test_matrix_input_keeps_its_shape():
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = mr.numerical_grad(A, lambda X: np.trace(X @ X))
        np.testing.assert_allclose(grad.reshape(2, 2), 2 * A.T, rtol=1e-8)


def test_matrix_valued_output_is_flattened_row_major():
        x = np.array([1.0, 2.0])
        jac = mr.numerical_grad(x, lambda v: np.outer(v, v))
        expected = np.array([[2 * x[0], 0], [x[1], x[0]], [x[1], x[0]], [0, 2 * x[1]]])
        np.testing.assert_allclose(jac, expected, rtol=1e-8, atol=1e-10)


def test_directional_derivative_is_linear_in_direction():
        f = lambda v: np.array([3 * v[0] + v[1] ** 2, np.sin(v[0])])
        x = np.array([0.4, 1.5])
        J = mr.numerical_grad(x, f)
        for direction in (np.array([1.0, 0.0]), np.array([2.0, -3.0])):
                np.testing.assert_allclose(mr.numerical_difference(x, f, direction=direction), J @ direction, rtol=1e-8)
        np.testing.assert_allclose(mr.numerical_difference(x, f), J @ np.ones(2), rtol=1e-8)


def test_numerical_grad_uses_update_func():
        R0 = mr.SO3.exp(np.array([0.2, -0.4, 0.6]))
        p = np.array([1.0, 2.0, 3.0])
        f = lambda w: R0 @ mr.SO3.exp(w) @ p
        # Body-frame perturbation: d(R0 exp(w^) p)/dw at w = 0 is -R0 hat(p).
        jac = mr.numerical_grad(np.zeros(3), f)
        np.testing.assert_allclose(jac, -R0 @ mr.SO3.hat(p), rtol=1e-8, atol=1e-10)
        # A custom update that doubles each step doubles the Jacobian.
        doubled = mr.numerical_grad(np.zeros(3), f, update_func=lambda w, d, offset: w + 2 * offset * d)
        np.testing.assert_allclose(doubled, 2 * jac, rtol=1e-8, atol=1e-10)


def test_complex_points_are_not_truncated():
        grad = mr.numerical_grad(np.array([1.0 + 1.0j]), lambda z: z ** 2)
        np.testing.assert_allclose(grad, [2.0 + 2.0j], rtol=1e-8)
