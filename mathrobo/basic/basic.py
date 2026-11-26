#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.06.23 Created by T.Ishigaki

import numpy as np
import jax.numpy as jnp

import math

__all__ = ['lib_error_message', 'iszero', 'sin', 'cos', 'zeros', 'identity', 'norm', 'isclose', 'gq_integrate', 'jac_lie_wrt_scaler', 'jac_adj_lie_wrt_scaler', 'jac_lie_v_wrt_vector']


def lib_error_message():
    return ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

def iszero(x):
    tolerance = 1e-8  # Tolerance
    return math.isclose(x, 0, abs_tol=tolerance)

def sin(theta, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.sin(theta)
    elif LIB == 'jax':
        return jnp.sin(theta)
    else:
        raise lib_error_message()
    
def cos(theta, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.cos(theta)
    elif LIB == 'jax':
        return jnp.cos(theta)
    else:
        raise lib_error_message()

def zeros(shape, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.zeros(shape)
    elif LIB == 'jax':
        return jnp.zeros(shape)
    else:
        raise lib_error_message()
    
def identity(size, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.identity(size)
    elif LIB == 'jax':
        return jnp.identity(size)
    else:
        raise lib_error_message()  

def norm(vec, LIB = 'numpy'):
        if LIB == 'numpy':
            return np.linalg.norm(vec)
        elif LIB == 'jax':
            return jnp.linalg.norm(vec)
        else:
            raise lib_error_message()
        
def isclose(a, b, LIB = 'numpy'):
        if LIB == 'numpy':
                return np.isclose(a, b)
        elif LIB == 'jax':
                return jnp.isclose(a, b)
        else:
                raise lib_error_message()
        
        
def gq_integrate(func, a, b, digit = 5, LIB = 'numpy'):
    """Integrate ``func`` over [a, b] with ``digit``-point Gauss-Legendre quadrature."""
    if LIB != 'numpy':
        raise ValueError("This method for 'numpy'")
    if not isinstance(digit, (int, np.integer)) or digit < 1:
        raise ValueError("digit must be a positive integer")
    nodes, weights = np.polynomial.legendre.leggauss(int(digit))
    half_width, center = 0.5 * (b - a), 0.5 * (a + b)
    integ = 0.0
    for x, w in zip(nodes, weights):
        integ = integ + (w * half_width) * np.asarray(func(x * half_width + center))
    return integ

def jac_lie_wrt_scaler(lie, vec, a, dvec, LIB = 'numpy'):
    m = lie.exp(vec, a, LIB)
    integ_m = -lie.exp_integ_adj(vec, -a, LIB)

    return m @ lie.hat(integ_m @ dvec, LIB)

def jac_adj_lie_wrt_scaler(lie, vec, a, dvec, LIB = 'numpy'):
    m = lie.exp_adj(vec, a, LIB)
    integ_m = -lie.exp_integ_adj(vec, -a, LIB)

    return m @ lie.hat_adj(integ_m @ dvec, LIB)

def jac_lie_v_wrt_vector(lie, vec, a, v, LIB = 'numpy'):
    m = lie.exp(vec, a, LIB)
    integ_m = -lie.exp_integ_adj(vec, -a, LIB)

    return m @ lie.hat_commute_adj(v, LIB) @ integ_m
