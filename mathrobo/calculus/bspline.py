import numpy as np
from scipy.interpolate import BSpline

# ----------------------------------------------------------------------
def build_bspline_model(knots, control, degree: int):
    """
    Parameters
    ----------
    knots   : (m+1,)   non-decreasing knot vector
    control : (n, D)   control-point matrix
    degree  : int      spline order k

    Returns
    -------
    curve(tq, order)  -> (..., D)
    jac  (tq, order)  -> (M*D, n*D)
    """
    T = np.asarray(knots, float)
    P = np.asarray(control, float)
    n_ctrl, D = P.shape
    k = degree

    # --- dimension-wise splines ---------------------------------------
    spl = [BSpline(T, P[:, d], k) for d in range(D)]

    # cache derivatives up to order k
    deriv_tbl = [[s.derivative(o) if o else s for o in range(k + 1)] for s in spl]

    # basis splines for Jacobian
    I = np.eye(n_ctrl)
    bases = [BSpline(T, I[i], k) for i in range(n_ctrl)]
    bases_der = [[b.derivative(o) if o else b for o in range(k + 1)]
                 for b in bases]

    def curve(tq, order: int = 0):
        if order < 0 or order > k:
            raise ValueError(f"order must be 0-{k}")
        tq = np.asarray(tq, float)
        return np.column_stack([tbl[order](tq) for tbl in deriv_tbl])

    def jac(tq, order: int = 0):
        if order < 0 or order > k:
            raise ValueError(f"order must be 0-{k}")
        tq = np.asarray(tq, float).ravel()
        M = tq.size

        # derivative of basis (M, n_ctrl)
        B = np.column_stack([bases_der[i][order](tq) for i in range(n_ctrl)])
        return np.kron(B, np.eye(D)).reshape(M * D, n_ctrl * D)

    return curve, jac

def bspline_curve(knots, control, degree, t, order=0):
    """
    B-spline curve at evaluation points.

    Parameters
    ----------
    knots : array_like, shape (m+1,)
        Non-decreasing knot vector.
    control : array_like, shape (n_ctrl, D)
        Control point matrix.
    degree : int
        Spline degree.
    t : array_like
        Parameter values at which to evaluate.
    order : int, optional
        Derivative order (0 means curve value). Must be 0 <= order <= degree.

    Returns
    -------
    curve_vals : ndarray, shape (len(t), D)
        Evaluated curve or derivative.
    """
    T = np.asarray(knots, float)
    P = np.asarray(control, float)
    _, D = P.shape
    k = degree

    if order < 0 or order > k:
        raise ValueError(f"order must be between 0 and {k}")

    t = np.asarray(t, float)
    result = []
    for d in range(D):
        spline = BSpline(T, P[:, d], k)
        if order > 0:
            spline = spline.derivative(order)
        result.append(spline(t))
    return np.column_stack(result)


def bspline_jacobian(knots, n_ctrl, degree, t, order=0):
    """
    Compute the Jacobian of the B-spline curve wrt control points.

    Parameters
    ----------
    knots : array_like, shape (m+1,)
        Non-decreasing knot vector.
    control : array_like, shape (n_ctrl, D)
        Control point matrix.
    degree : int
        Spline degree.
    t : array_like
        Parameter values at which to evaluate.
    order : int, optional
        Derivative order. Must be 0 <= order <= degree.

    Returns
    -------
    J : ndarray, shape (len(t)*D, n_ctrl*D)
        Jacobian matrix d(Curve)/d(control).
    """
    T = np.asarray(knots, float)
    k = degree

    if order < 0 or order > k:
        raise ValueError(f"order must be between 0 and {k}")

    t = np.asarray(t, float).ravel()
    I = np.eye(n_ctrl)
    bases = [BSpline(T, I[i], k) for i in range(n_ctrl)]
    B = np.column_stack([
        (bases[i].derivative(order) if order > 0 else bases[i])(t)
        for i in range(n_ctrl)
    ])  # shape (len(t), n_ctrl)

    J = B
    return J



# ----------------------------------------------------------------------
def jac_numerical(curve, ctrl, knots, k, tq, h=1e-6, order=0):
    """
    Finite-difference Jacobian for validation
    curve(tq, order)  : 0-k 階(k = degree)まで評価
    ctrl              : (n_ctrl, D)   control-point matrix
    knots             : (m+1,)        non-decreasing knot vector
    k                 : int           spline order k
    tq                : (M,)         evaluation points
    h                 : float         finite-difference step size
    order            : int           derivative order (0-k)
    """
    n_ctrl, D = ctrl.shape
    M = len(tq)
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