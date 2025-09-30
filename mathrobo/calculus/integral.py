import math
import numpy as np

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
                Integration method: "euler", "poly", "rk2", "rk4".

        Returns
        -------
        mat : ndarray of shape (dof * order, dof * order)
                State transition matrix.
        vec : ndarray of shape (dof * order, dof)
                Control/integration vector matrix (for highest-order derivative input).
        """
        dim = dof * order
        A = np.eye(dim)
        B = np.zeros((dim, dof))

        if method == "euler":
                # 1st-order approximation: x_{k+1} ≈ x_k + dt * dx_k
                for i in range(order - 1):
                        A[i * dof:(i + 1) * dof, (i + 1) * dof:(i + 2) * dof] = dt * np.eye(dof)
                B[(order - 1) * dof:] = dt * np.eye(dof)

        elif method == "poly":
                # Arbitrary-order Taylor expansion: dt^n / n!
                for i in range(1, order):
                        for j in range(i):
                                power = i - j
                                coef = (dt ** power) / math.factorial(power)
                                A[j*dof:(j+1)*dof, i*dof:(i+1)*dof] = coef * np.eye(dof)
                for i in range(order):
                        power = order - i
                        coef = (dt ** power) / math.factorial(power)
                        B[i*dof:(i+1)*dof] = coef * np.eye(dof)

        elif method == "rk2":
                # Midpoint method approximation (for 2nd-order accuracy)
                # x(t+dt) ≈ x(t) + dt * f(x(t) + 0.5*dt*f(x))
                for i in range(1, order):
                        for j in range(i):
                                coef = dt ** (i - j)
                                if i - j == 1:
                                        coef += 0.5 * dt ** (i - j)
                                A[j*dof:(j+1)*dof, i*dof:(i+1)*dof] = coef * np.eye(dof)

                for i in range(order):
                        coef = dt ** (order - i)
                        if order - i == 1:
                                coef += 0.5 * dt ** (order - i)
                        B[i*dof:(i+1)*dof] = coef * np.eye(dof)

        elif method == "rk4":
                # RK4 is not straightforwardly expressed in this matrix form,
                # but we can approximate with higher-weighted Taylor expansion
                coeffs = [1, 1/2, 1/6]  # leading coefficients for 1st, 2nd, 3rd derivatives
                for i in range(1, order):
                        for j in range(i):
                                power = i - j
                                if power <= len(coeffs):
                                        coef = coeffs[power - 1] * dt ** power
                                        A[j*dof:(j+1)*dof, i*dof:(i+1)*dof] = coef * np.eye(dof)

                for i in range(order):
                        power = order - i
                        if power <= len(coeffs):
                                coef = coeffs[power - 1] * dt ** power
                                B[i*dof:(i+1)*dof] = coef * np.eye(dof)

        else:
                raise ValueError(f"Unsupported method: {method}")

        return A, B
