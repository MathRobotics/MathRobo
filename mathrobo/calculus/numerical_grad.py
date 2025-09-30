#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.05.08 Created by T.Ishigaki

import numpy as np

def _finite_difference(func, x, eps=1e-8, method="central", sub_func=None, update_func = None, direction=None):
        """
        Unified finite difference backend to compute either:
        - full Jacobian (if direction is None)
        - directional derivative (if direction is specified)

        Parameters
        ----------
        func : callable
                Function R^n -> R^m
        x : ndarray
                Point at which to evaluate
        eps : float
                Perturbation step
        method : str
                'forward', 'central', or 'fourth'
        sub_func : callable or None
                Custom difference function: sub_func(y0, y1) = y1 - y0
        direction : ndarray or None
                If provided, compute derivative along this direction vector

        Returns
        -------
        grad : ndarray
                Jacobian (m x n) or directional derivative (m,)
        """
        x = np.asarray(x, dtype=float).flatten()

        if sub_func is None:
                y0 = np.atleast_1d(func(x))
                y_dim = y0.size
        else:
                y0 = func(x)
                y_dim = sub_func(y0, y0).size
        
        x_dim = x.size

        def update(x, dircetion, offset):
                if update_func is None:
                        return x + offset*dircetion
                else:
                        return update_func(x, dircetion, offset)

        def eval(dircetion, offset):
                if sub_func is None:
                        return np.atleast_1d(func(update(x, dircetion, offset)))
                else:
                        return func(update(x, dircetion, offset))

        def diff_eval(dircetion, eps):
                if method == "forward":
                        y1 = eval(dircetion, eps)
                        dy = sub_func(y0, y1) if sub_func else y1 - y0
                        return dy / eps

                elif method == "central":
                        y1 = eval(dircetion, -eps)
                        y2 = eval(dircetion, +eps)
                        dy = sub_func(y1, y2) if sub_func else y2 - y1
                        return dy / (2 * eps)

                elif method == "fourth":
                        y_m2h = eval(dircetion, -2 * eps)
                        y_m1h = eval(dircetion, -1 * eps)
                        y_p1h = eval(dircetion, +1 * eps)
                        y_p2h = eval(dircetion, +2 * eps)
                        if sub_func:
                                t1 = 8 * sub_func(y_m1h, y_p1h)
                                t2 = sub_func(y_m2h, y_p2h)
                                dy = (t1 - t2) / (12 * eps)
                        else:
                                dy = (-y_p2h + 8*y_p1h - 8*y_m1h + y_m2h) / (12 * eps)
                        return dy

                else:
                        raise ValueError(f"Unsupported method: {method}")

        if direction is None:
                #compute full Jacobian (each input dimension perturbed independently)
                grad = np.zeros((y_dim, x_dim))

                for i in range(x_dim):
                        ei = np.zeros_like(x)
                        ei[i] = 1.0

                        grad[:, i]  = diff_eval(ei, eps)

                return grad if y_dim > 1 else grad.flatten()
        else:
                # direction is given: compute directional derivative
                direction = np.asarray(direction, dtype=float).flatten()
                norm = np.linalg.norm(direction)
                if norm == 0:
                        raise ValueError("Direction vector must not be zero.")
                direction = direction / norm

                return diff_eval(direction, eps)

def numerical_grad(x, func, eps=1e-8, method="central", sub_func=None):
        return _finite_difference(func, x, eps=eps, method=method, sub_func=sub_func, direction=None)

def numerical_difference(x, func, eps=1e-8, method="central", sub_func=None, update_func=None, direction=None):
        if direction is None:
                direction = np.ones_like(x)
        return _finite_difference(func, x, eps=eps, method=method, sub_func=sub_func, update_func=update_func, direction=direction)
