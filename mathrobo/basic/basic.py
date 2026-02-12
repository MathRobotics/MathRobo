#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-
# 2024.06.23 Created by T.Ishigaki

import numpy as np
import sympy as sp
import jax.numpy as jnp

import math

def lib_error_message():
    ValueError("Unsupported library. Choose 'numpy', 'sympy' or 'jax'.")

def iszero(x):
    tolerance = 1e-8  # 許容範囲
    return math.isclose(x, 0, abs_tol=tolerance)

def sin(theta, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.sin(theta)
    elif LIB == 'sympy':
        return sp.sin(theta)
    elif LIB == 'jax':
        return jnp.sin(theta)
    else:
        raise lib_error_message()
    
def cos(theta, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.cos(theta)
    elif LIB == 'sympy':
        return sp.cos(theta)
    elif LIB == 'jax':
        return jnp.cos(theta)
    else:
        raise lib_error_message()

def zeros(shape, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.zeros(shape)
    elif LIB == 'sympy':
        if type(shape) == int:
            return sp.zeros(shape,1)
        elif type(shape) == tuple and len(shape) == 2:
            return sp.zeros(shape[0],shape[1])
    elif LIB == 'jax':
        return jnp.zeros(shape)
    else:
        raise lib_error_message()
    
def identity(size, LIB = 'numpy'):
    if LIB == 'numpy':
        return np.identity(size)
    elif LIB == 'sympy':
        m = sp.zeros(size,size)
        for i in range(size):
            m[i,i] = 1
        return m
    elif LIB == 'jax':
        return jnp.identity(size)
    else:
        raise lib_error_message()  

def norm(vec, LIB = 'numpy'):
        if LIB == 'numpy':
            return np.linalg.norm(vec)
        elif LIB == 'sympy':
            return sp.sqrt(vec.dot(vec))
        elif LIB == 'jax':
            return jnp.linalg.norm(vec)
        else:
            raise lib_error_message()
        
def isclose(a, b, LIB = 'numpy'):
        if LIB == 'numpy':
                return np.isclose(a, b)
        elif LIB == 'sympy':
                return sp.simplify(a - b) == 0
        elif LIB == 'jax':
                return jnp.isclose(a, b)
        else:
                raise lib_error_message()
        
        
def gq_integrate(func, a, b, digit = 5, LIB = 'numpy'):
        if LIB == 'numpy':
            integ = np.zeros((func(0).shape))

            if digit == 3:
                quad_x = np.array((-0.77459666924, 0, 0.77459666924))
                quad_weight = np.array((0.55555555555, 0.88888888888, 0.55555555555))
            elif digit == 4:
                quad_x = np.array((-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116))
                quad_weight = np.array((0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451))
            elif digit == 5:
                quad_x = np.array((-0.9061798459, -0.5384693101, 0, 0.5384693101, 0.9061798459))
                quad_weight = np.array((0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851))
            
            for i in range(digit):
                w = quad_weight[i] * (b - a) * 0.5
                x = quad_x[i] * (b - a) * 0.5 + (a + b) * 0.5
                integ += w*func(x)

            return integ
        else:
            raise ValueError("This method for 'numpy'")

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

def sympy_to_numpy(sp_mat):
    return np.array(sp_mat).astype(np.float64)

def sympy_subs_mat(m, vec_str, vec_val):
    for i in range(len(vec_str)):
        m = m.subs([(vec_str[i], vec_val[i])])

    return m
