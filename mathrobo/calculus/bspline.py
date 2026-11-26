import numpy as np
from scipy.interpolate import BSpline

__all__ = ['build_bspline_model', 'bspline_curve', 'bspline_jacobian', 'jac_numerical']

# All functions evaluate on the base interval [knots[degree], knots[n_ctrl]]
# and return NaN outside it, so curves and their Jacobians always agree.


def _check_degree(degree):
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 0:
                raise ValueError("degree must be a non-negative integer")
        return int(degree)


def _check_order(order, degree):
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= degree:
                raise ValueError(f"order must be between 0 and {degree}")
        return int(order)


def _knots(knots, n_ctrl, degree):
        T = np.asarray(knots, dtype=np.float64)
        if T.ndim != 1:
                raise ValueError("knots must be a 1-D array")
        if np.any(np.diff(T) < 0):
                raise ValueError("knots must be non-decreasing")
        if T.size != n_ctrl + degree + 1:
                raise ValueError(
                        f"len(knots) must equal n_ctrl + degree + 1 = {n_ctrl + degree + 1}, got {T.size}")
        return T


def _control(control):
        P = np.asarray(control, dtype=np.float64)
        if P.ndim != 2:
                raise ValueError("control must have shape (n_ctrl, D)")
        return P


def _points(t):
        return np.asarray(t, dtype=np.float64).reshape(-1)


def _basis(T, n_ctrl, degree):
        """All basis functions at once: evaluating gives shape (M, n_ctrl)."""
        return BSpline(T, np.eye(n_ctrl), degree, extrapolate=False)


def _derivative(spline, order):
        return spline.derivative(order) if order else spline


# ----------------------------------------------------------------------
def build_bspline_model(knots, control, degree: int):
        """
        Parameters
        ----------
        knots   : (n_ctrl + degree + 1,)   non-decreasing knot vector
        control : (n_ctrl, D)              control-point matrix
        degree  : int                      spline degree k

        Returns
        -------
        curve(tq, order)  -> (M, D)          with M = tq.size
        jac  (tq, order)  -> (M*D, n_ctrl*D) d vec(curve) / d vec(control), row-major
        """
        k = _check_degree(degree)
        P = _control(control)
        n_ctrl, D = P.shape
        T = _knots(knots, n_ctrl, k)

        spline = BSpline(T, P, k, extrapolate=False)
        basis = _basis(T, n_ctrl, k)
        # cache derivatives up to order k
        curves = [_derivative(spline, o) for o in range(k + 1)]
        bases = [_derivative(basis, o) for o in range(k + 1)]
        eye = np.eye(D)

        def curve(tq, order: int = 0):
                return curves[_check_order(order, k)](_points(tq))

        def jac(tq, order: int = 0):
                return np.kron(bases[_check_order(order, k)](_points(tq)), eye)

        return curve, jac


def bspline_curve(knots, control, degree, t, order=0):
        """
        B-spline curve at evaluation points.

        Parameters
        ----------
        knots : array_like, shape (n_ctrl + degree + 1,)
                Non-decreasing knot vector.
        control : array_like, shape (n_ctrl, D)
                Control point matrix.
        degree : int
                Spline degree.
        t : array_like
                Parameter values at which to evaluate (flattened).
        order : int, optional
                Derivative order (0 means curve value). Must be 0 <= order <= degree.

        Returns
        -------
        curve_vals : ndarray, shape (t.size, D)
                Evaluated curve or derivative; NaN outside the base interval.
        """
        k = _check_degree(degree)
        P = _control(control)
        T = _knots(knots, P.shape[0], k)
        order = _check_order(order, k)
        return _derivative(BSpline(T, P, k, extrapolate=False), order)(_points(t))


def bspline_jacobian(knots, n_ctrl, degree, t, order=0):
        """
        Basis matrix B with ``bspline_curve(knots, control, degree, t, order) == B @ control``.

        Parameters
        ----------
        knots : array_like, shape (n_ctrl + degree + 1,)
                Non-decreasing knot vector.
        n_ctrl : int
                Number of control points.
        degree : int
                Spline degree.
        t : array_like
                Parameter values at which to evaluate (flattened).
        order : int, optional
                Derivative order. Must be 0 <= order <= degree.

        Returns
        -------
        B : ndarray, shape (t.size, n_ctrl)
                Derivatives of the basis functions; NaN outside the base interval.
                It is shared by every dimension of the control points.
        """
        k = _check_degree(degree)
        if isinstance(n_ctrl, bool) or not isinstance(n_ctrl, (int, np.integer)) or n_ctrl < 1:
                raise ValueError("n_ctrl must be a positive integer")
        T = _knots(knots, int(n_ctrl), k)
        order = _check_order(order, k)
        return _derivative(_basis(T, int(n_ctrl), k), order)(_points(t))


# ----------------------------------------------------------------------
def jac_numerical(curve, ctrl, knots, k, tq, h=1e-6, order=0):
        """
        Finite-difference Jacobian for validation
        curve(tq, order)  : evaluate derivatives of orders 0 through k (k = degree)
        ctrl              : (n_ctrl, D)   control-point matrix
        knots             : (n_ctrl + k + 1,) non-decreasing knot vector
        k                 : int           spline degree
        tq                : (M,)         evaluation points
        h                 : float         finite-difference step size
        order            : int           derivative order (0-k)
        """
        ctrl = _control(ctrl)
        n_ctrl, D = ctrl.shape
        M = _points(tq).size
        J_fd = np.empty((M * D, n_ctrl * D))

        base = curve(tq, order).reshape(M * D)

        for i in range(n_ctrl):
                for d in range(D):
                        pert = ctrl.copy()
                        pert[i, d] += h
                        curve_p, _ = build_bspline_model(knots, pert, k)
                        col = (curve_p(tq, order).reshape(M * D) - base) / h
                        J_fd[:, i * D + d] = col
        return J_fd
