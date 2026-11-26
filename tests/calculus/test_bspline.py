#!/usr/bin/env python3

import numpy as np
import mathrobo as mr

# ----------------------------------------------------------------------
def run_single_test(n_ctrl=10, D=3, k=5, M=100, tol=1e-4, seed=None):
        """
        Compare analytic vs finite-diff Jacobian for order=0..k
        n_ctrl : int  number of control points
        D      : int  dimension of control points
        k      : int  degree of spline (order = k)
        M      : int  number of evaluation points
        tol    : float  tolerance for finite-diff Jacobian
        seed   : int  random seed for control points and knots
        """
        print(f"n_ctrl={n_ctrl}, D={D}, k={k}, M={M}, tol={tol}")
        rng = np.random.default_rng(seed)
        ctrl  = rng.standard_normal((n_ctrl, D))
        # knots = np.linspace(0, 1, M)
        knots = np.concatenate((np.zeros(k),
                np.linspace(0, 1, n_ctrl - k + 1),
                np.ones(k)))

        curve, jac = mr.build_bspline_model(knots, ctrl, k)
        tq = np.sort(rng.random(M))

        for order in range(k+1):
                J_a = jac(tq, order)
                J_n = mr.jac_numerical(curve, ctrl, knots, k, tq, order=order)
                print(f"order {order}: J_a.shape = {J_a.shape}, J_n.shape = {J_n.shape}")
                err = np.linalg.norm(J_a - J_n, np.inf)
                print(f"order {order}: max|ΔJ| = {err:.2e}")
                assert err < tol, f"Jacobian failed at order {order}"
        print("✓ All orders passed")

        for order in range(k+1):
                p = curve(tq, order)
                jacob = jac(tq, order)
                p_jac = jacob @ ctrl.flatten()
                p_jac = p_jac.reshape(M, D)
                err = np.linalg.norm(p - p_jac, np.inf)
                print(f"order {order}: max|Δp| = {err:.2e}")
                assert err < tol, f"Jacobian failed at order {order}"
        print("✓ All orders passed")

        for order in range(k+1):
                p = mr.bspline_curve(knots, ctrl, k, tq, order=order)
                jacob = mr.bspline_jacobian(knots, n_ctrl, k, tq, order=order)
                p_jac = jacob @ ctrl
                err = np.linalg.norm(p - p_jac, np.inf)
                print(f"order {order}: max|Δp| = {err:.2e}")
                assert err < tol, f"Jacobian failed at order {order}"
        print("✓ All orders passed")

# ----------------------------------------------------------------------
def test_bspline():
    run_single_test(tol = 1e-3, k = 4)


import pytest
from scipy.interpolate import BSpline

K, N_CTRL, DIM = 3, 7, 2
KNOTS = np.concatenate((np.zeros(K), np.linspace(0, 1, N_CTRL - K + 1), np.ones(K)))
CTRL = np.random.default_rng(0).standard_normal((N_CTRL, DIM))


def test_inputs_may_be_lists_and_integer_arrays():
        t = [0.1, 0.5, 0.9]
        expected = mr.bspline_curve(KNOTS, np.arange(14.0).reshape(7, 2), K, np.array(t))
        np.testing.assert_allclose(mr.bspline_curve(list(KNOTS), np.arange(14).reshape(7, 2), K, t), expected)
        np.testing.assert_allclose(mr.bspline_jacobian(list(KNOTS), N_CTRL, K, t),
                                   mr.bspline_jacobian(KNOTS, N_CTRL, K, np.array(t)))


def test_jacobian_matches_scipy_design_matrix_and_curve():
        t = np.linspace(0, 1, 41)
        B = mr.bspline_jacobian(KNOTS, N_CTRL, K, t)
        np.testing.assert_allclose(B, BSpline.design_matrix(t, KNOTS, K).toarray(), atol=1e-14)
        for order in range(K + 1):
                B = mr.bspline_jacobian(KNOTS, N_CTRL, K, t, order=order)
                np.testing.assert_allclose(B @ CTRL, mr.bspline_curve(KNOTS, CTRL, K, t, order=order), atol=1e-10)


@pytest.mark.parametrize("order", [1, 2])
def test_derivatives_match_parameter_differences(order):
        t, h = np.array([0.23, 0.61]), 1e-4
        values = [mr.bspline_curve(KNOTS, CTRL, K, t + s * h, order=order - 1) for s in (-1, 1)]
        np.testing.assert_allclose(mr.bspline_curve(KNOTS, CTRL, K, t, order=order),
                                   (values[1] - values[0]) / (2 * h), rtol=1e-6, atol=1e-6)


def test_model_curve_and_jacobian_agree_inside_and_are_nan_outside():
        curve, jac = mr.build_bspline_model(KNOTS, CTRL, K)
        inside, outside = np.array([0.0, 0.4, 1.0]), np.array([-0.1, 1.2])
        for order in range(K + 1):
                np.testing.assert_allclose(jac(inside, order) @ CTRL.ravel(), curve(inside, order).ravel(), atol=1e-10)
                np.testing.assert_allclose(curve(inside, order), mr.bspline_curve(KNOTS, CTRL, K, inside, order))
        assert np.isnan(curve(outside)).all()
        assert np.isnan(jac(outside)).all()
        assert np.isnan(mr.bspline_curve(KNOTS, CTRL, K, outside)).all()


@pytest.mark.parametrize("n_ctrl", [N_CTRL - 1, N_CTRL + 1])
def test_knot_and_control_counts_must_agree(n_ctrl):
        control = np.zeros((n_ctrl, DIM))
        with pytest.raises(ValueError, match="len\\(knots\\)"):
                mr.bspline_curve(KNOTS, control, K, [0.5])
        with pytest.raises(ValueError, match="len\\(knots\\)"):
                mr.bspline_jacobian(KNOTS, n_ctrl, K, [0.5])
        with pytest.raises(ValueError, match="len\\(knots\\)"):
                mr.build_bspline_model(KNOTS, control, K)


@pytest.mark.parametrize("bad", [
        dict(control=np.zeros(N_CTRL)),
        dict(knots=KNOTS[::-1]),
        dict(degree=-1),
        dict(degree=2.0),
        dict(order=K + 1),
        dict(order=-1),
])
def test_invalid_arguments_are_rejected(bad):
        args = dict(knots=KNOTS, control=CTRL, degree=K, t=[0.5], order=0) | bad
        with pytest.raises(ValueError):
                mr.bspline_curve(**args)
