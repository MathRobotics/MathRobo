#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.05.08 Created by T.Ishigaki

import numpy as np

__all__ = ['numerical_grad', 'numerical_difference']

# Error order p of each stencil. The default step eps_mach**(1/(p+1)) balances
# truncation (~h**p) against rounding (~eps_mach/h) for well-scaled functions.
_ORDERS = {"forward": 1, "central": 2, "fourth": 4}


def _default_eps(method):
        if method not in _ORDERS:
                raise ValueError(f"Unsupported method: {method}")
        return float(np.finfo(np.float64).eps ** (1.0 / (_ORDERS[method] + 1)))


def _finite_difference(func, x, eps=None, method="central", sub_func=None, update_func=None, direction=None):
        """
        Unified finite difference backend to compute either:
        - full Jacobian (if direction is None)
        - directional derivative J @ direction (if direction is specified)

        Parameters
        ----------
        func : callable
                Function evaluated at arrays shaped like ``x``.
        x : array_like
                Point at which to evaluate. Its shape is preserved for ``func``.
        eps : float or None
                Perturbation step; None picks a step suited to ``method``.
        method : str
                'forward', 'central', or 'fourth'
        sub_func : callable or None
                Custom difference function: sub_func(y0, y1) = y1 - y0
        update_func : callable or None
                Custom perturbation: update_func(x, direction, offset), default
                ``x + offset * direction``. ``direction`` is shaped like ``x``.
        direction : array_like or None
                If provided, compute the derivative along this vector (not
                normalized, so the result scales linearly with it).

        Returns
        -------
        grad : ndarray
                Jacobian (m x n) with flattened output and input, or (n,) for
                a single output; directional derivative shaped like the output.
        """
        if eps is None:
                eps = _default_eps(method)
        elif method not in _ORDERS:
                raise ValueError(f"Unsupported method: {method}")

        x = np.asarray(x)
        x = x.astype(np.result_type(x.dtype, np.float64), copy=False)

        def at(step_direction, offset):
                if update_func is None:
                        y = func(x + offset * step_direction)
                else:
                        y = func(update_func(x, step_direction, offset))
                return y if sub_func is not None else np.atleast_1d(np.asarray(y))

        def difference(y_from, y_to):
                return sub_func(y_from, y_to) if sub_func is not None else y_to - y_from

        base = []  # forward differences share one evaluation at x

        def diff_eval(step_direction):
                if method == "forward":
                        if not base:
                                base.append(at(np.zeros_like(x), 0.0))
                        return difference(base[0], at(step_direction, eps)) / eps
                if method == "central":
                        return difference(at(step_direction, -eps), at(step_direction, eps)) / (2 * eps)
                near = difference(at(step_direction, -eps), at(step_direction, eps))
                far = difference(at(step_direction, -2 * eps), at(step_direction, 2 * eps))
                return (8 * near - far) / (12 * eps)

        if direction is not None:
                direction = np.asarray(direction, dtype=x.dtype)
                if direction.size != x.size:
                        raise ValueError("Direction must have as many elements as x.")
                return np.asarray(diff_eval(direction.reshape(x.shape)))

        columns = []
        for i in range(x.size):
                unit = np.zeros(x.size, dtype=x.dtype)
                unit[i] = 1.0
                columns.append(np.asarray(diff_eval(unit.reshape(x.shape))).ravel())
        grad = np.stack(columns, axis=-1) if columns else np.zeros((0, 0))
        return grad if grad.shape[0] > 1 else grad.reshape(-1)


def numerical_grad(x, func, eps=None, method="central", sub_func=None, update_func=None):
        """Finite-difference Jacobian of ``func`` at ``x`` (see ``_finite_difference``)."""
        return _finite_difference(func, x, eps=eps, method=method, sub_func=sub_func,
                                  update_func=update_func, direction=None)


def numerical_difference(x, func, eps=None, method="central", sub_func=None, update_func=None, direction=None):
        """Finite-difference directional derivative ``J @ direction`` (default: all ones)."""
        if direction is None:
                direction = np.ones_like(np.asarray(x, dtype=float))
        return _finite_difference(func, x, eps=eps, method=method, sub_func=sub_func,
                                  update_func=update_func, direction=direction)
