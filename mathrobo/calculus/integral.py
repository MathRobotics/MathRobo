import math
import numpy as np

__all__ = ['build_integrator']


def build_integrator(dof, order, dt, method="euler"):
        """
        Construct integration matrices for various numerical integration methods.

        Parameters
        ----------
        dof : int
                Degrees of freedom.
        order : int
                Order of the state (e.g., 2 for [q, dq], 3 for [q, dq, ddq]).
        dt : float
                Time step.
        method : str
                "euler", "rk2" or "rk4" (explicit Runge-Kutta of order 1, 2 or 4),
                or "poly" (the exact discretization for a constant input).

        Returns
        -------
        mat : ndarray of shape (dof * order, dof * order)
                State transition matrix.
        vec : ndarray of shape (dof * order, dof)
                Control/integration vector matrix (for highest-order derivative input).
        """
        # For the integrator chain x' = S x + E u with constant u, an explicit
        # Runge-Kutta method of order p reproduces the Taylor series of the
        # exact discretization up to dt**p, so every method is a truncation.
        max_power = {"euler": 1, "rk2": 2, "rk4": 4, "poly": order}.get(method)
        if max_power is None:
                raise ValueError(f"Unsupported method: {method}")

        dim = dof * order
        A = np.eye(dim)
        B = np.zeros((dim, dof))
        eye = np.eye(dof)

        # Only powers up to max_power are nonzero, so fill just those block
        # diagonals: O(order * max_power) blocks instead of O(order**2).
        for power in range(1, min(max_power, order) + 1):
                block = dt ** power / math.factorial(power) * eye
                for j in range(order - power):
                        i = j + power
                        A[j*dof:(j+1)*dof, i*dof:(i+1)*dof] = block
                i = order - power
                B[i*dof:(i+1)*dof] = block

        return A, B
