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
