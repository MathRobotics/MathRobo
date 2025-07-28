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
        knots = np.linspace(0, 1, M)

        curve, jac = mr.build_bspline_model(knots, ctrl, k)
        tq = np.sort(rng.random(M))

        print(tq)

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
                J = jac(tq, order)
                p_jac = J @ ctrl.flatten()
                p_jac = p_jac.reshape(M, D)
                err = np.linalg.norm(p - p_jac, np.inf)
                print(f"order {order}: max|Δp| = {err:.2e}")
                assert err < tol, f"Jacobian failed at order {order}"
        print("✓ All orders passed")

        for order in range(k+1):
                p = mr.bspline_curve(knots, ctrl, k, tq, order=order)
                J = mr.bspline_jacobian(knots, n_ctrl, k, tq, order=order)
                p_jac = J @ ctrl
                err = np.linalg.norm(p - p_jac, np.inf)
                print(f"order {order}: max|Δp| = {err:.2e}")
                assert err < tol, f"Jacobian failed at order {order}"
        print("✓ All orders passed")


# ----------------------------------------------------------------------
if __name__ == "__main__":
        run_single_test(tol = 1e-3, k = 4)
