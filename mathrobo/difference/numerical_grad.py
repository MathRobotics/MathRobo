#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2025.05.08 Created by T.Ishigaki

import numpy as np

def numerical_grad(x, func, eps=1e-8, method="central"):
    x = np.asarray(x, dtype=float).flatten()
    y0 = np.atleast_1d(func(x))
    grad = np.zeros((y0.size, x.size))

    for i in range(x.size):
        x1 = x.copy()
        if method == "forward":
            x1[i] += eps
            y1 = np.atleast_1d(func(x1))
            grad[:, i] = (y1 - y0) / eps

        elif method == "central":
            x1[i] += eps
            x2 = x.copy()
            x2[i] -= eps
            y1 = np.atleast_1d(func(x1))
            y2 = np.atleast_1d(func(x2))
            grad[:, i] = (y1 - y2) / (2 * eps)

        elif method == "fourth":
            # 4th-order central difference: f(x - 2h), f(x - h), f(x + h), f(x + 2h)
            x_m2h = x.copy(); x_m2h[i] -= 2 * eps
            x_m1h = x.copy(); x_m1h[i] -= 1 * eps
            x_p1h = x.copy(); x_p1h[i] += 1 * eps
            x_p2h = x.copy(); x_p2h[i] += 2 * eps
            f_m2h = np.atleast_1d(func(x_m2h))
            f_m1h = np.atleast_1d(func(x_m1h))
            f_p1h = np.atleast_1d(func(x_p1h))
            f_p2h = np.atleast_1d(func(x_p2h))
            grad[:, i] = (-f_p2h + 8*f_p1h - 8*f_m1h + f_m2h) / (12 * eps)

        else:
            raise ValueError(f"Unsupported method: {method}")

    return grad if y0.size > 1 else grad.flatten()
