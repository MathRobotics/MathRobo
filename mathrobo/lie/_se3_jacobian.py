"""Coefficients of the unit adjoint integral and their angle derivatives.

For SE(3), K = ad_xi has a minimal polynomial dividing K (K² + s I)², with s
the squared rotation angle. For SO(3), K = hat(phi) satisfies K³ = -s K, so
the same four-coefficient polynomial applies to 3 x 3 generators unchanged.
"""

from functools import lru_cache
from math import comb, factorial, sin, sqrt

import numpy as np

from .._taylor import taylor_matmul, taylor_matmul_left_vjp, taylor_matmul_vjp


@lru_cache(maxsize=8)
def _series_table(count):
    # Polynomials in s for the normalized derivatives A^(n)(s)/n! and B^(n)(s)/n!.
    return np.array([
        [[(-1)**(degree + n) * comb(degree + n, n) / factorial(2 * (degree + n) + offset)
          for offset in (2, 3)] for n in range(count)]
        for degree in range(13)
    ])


_SCALAR_SERIES = _series_table(4)[::-1].transpose(1, 2, 0).tolist()


def _regular_terms(s, order, sine, square_root):
    theta = square_root(s)
    sinc = sine(theta) / theta
    a = 2 * (sine(theta / 2) / theta)**2
    b = (1 - sinc) / s
    terms = [(a, b)]
    # (s A)' = sinc/2, (s B)' = -sinc', and sinc' = (B - A)/2.
    for n in range(1, order + 2):
        next_sinc = (b - a) / (2 * n)
        a = (sinc / (2 * n) - a) / s
        b = (-next_sinc - b) / s
        terms.append((a, b))
        sinc = next_sinc
    return terms


def _scalar_polynomial(s, coefficients):
    value = coefficients[0]
    for coefficient in coefficients[1:]:
        value = value * s + coefficient
    return value


def _batched_terms(s, order, xp):
    # Both branches are evaluated by where. Keep their inactive arguments safe.
    small = xp.abs(s) <= 4
    regular_s = xp.where(small, xp.ones_like(s), s)
    regular = _regular_terms(regular_s, order, xp.sin, xp.sqrt)
    polynomial = xp.asarray(_series_table(max(4, order + 2))[:, :order + 2], dtype=s.dtype)
    series_s = xp.where(small, s, xp.zeros_like(s))[..., None, None]
    series = polynomial[-1]
    for term in polynomial[-2::-1]:
        series = series * series_s + term
    regular = xp.stack([xp.stack(pair, axis=-1) for pair in regular], axis=-2)
    terms = xp.where(small[..., None, None], series, regular)
    return [(terms[..., n, 0], terms[..., n, 1]) for n in range(order + 2)]


def adjoint_integral_coefficients(s, order, xp, *, normalized=False):
    """Return derivatives through ``order`` of the four polynomial coefficients."""
    scalar = xp is np and order <= 2 and s.ndim == 0 and s.dtype.kind == 'f' and s.dtype.itemsize <= 8
    if scalar:
        angle_squared = s.item()
        if angle_squared <= 4:
            terms = [(_scalar_polynomial(angle_squared, a), _scalar_polynomial(angle_squared, b))
                     for a, b in _SCALAR_SERIES[:order + 2]]
        else:
            terms = _regular_terms(angle_squared, order, sin, sqrt)
    else:
        angle_squared = s
        terms = _batched_terms(s, order, xp)

    # For K = ad_xi, the minimal polynomial divides K (K² + s I)².
    # The integral is I + (A - s A') K + (B - s B') K² - A' K³ - B' K⁴.
    coefficients = []
    for n in range(order + 1):
        a, b = terms[n]
        da, db = terms[n + 1]
        da, db = (n + 1) * da, (n + 1) * db
        row = ((1 - n) * a - angle_squared * da, (1 - n) * b - angle_squared * db, -da, -db)
        if not normalized and n > 1:
            row = tuple(factorial(n) * entry for entry in row)
        coefficients.append(np.asarray(row, dtype=s.dtype) if scalar else row)
    return coefficients


def _adjoint_polynomial_series(angular, xp, *, derivative=False):
    """Return the four scalar coefficients and optionally their derivatives in s."""
    order = angular.shape[-2]
    angular_row = angular[..., None, :]
    squared = taylor_matmul(angular_row, angular_row.swapaxes(-1, -2), xp)
    base = squared[..., 0, 0, 0]
    derivatives = adjoint_integral_coefficients(
        base, order if derivative else order - 1, xp, normalized=True,
    )
    derivatives = xp.stack([xp.stack(row, axis=-1) for row in derivatives], axis=-2)

    # Compose the scalar coefficient functions with s(t) about s(0).
    one = xp.concatenate((xp.ones_like(squared[..., :1, :, :]),
                          xp.zeros_like(squared[..., 1:, :, :])), axis=-3)
    delta_over_t = squared[..., 1:, :, :]
    power = one
    powers = [one[..., 0, 0]]
    for degree in range(1, order):
        count = order - degree
        power = taylor_matmul(power[..., :count, :, :], delta_over_t[..., :count, :, :], xp)
        zeros = xp.zeros_like(squared[..., :degree, 0, 0])
        powers.append(xp.concatenate((zeros, power[..., 0, 0]), axis=-1))
    composition = xp.stack(powers, axis=-1)
    coefficients = composition @ derivatives[..., :order, :]
    if not derivative:
        return coefficients, one

    scales = xp.arange(1, order + 1, dtype=derivatives.dtype)
    rates = composition @ (derivatives[..., 1:, :] * scales[:, None])
    return coefficients, rates


def _adjoint_polynomial_stages(coefficients, generator, xp):
    """Return the Horner stages before the final product with the generator."""
    identity = xp.eye(generator.shape[-1], dtype=generator.dtype)
    stages = [coefficients[..., 3, None, None] * identity]
    for i in range(2, -1, -1):
        stages.append(
            taylor_matmul(stages[-1], generator, xp) + coefficients[..., i, None, None] * identity
        )
    return stages


def adjoint_integral_series(angular, generator, xp):
    """Lift the unit adjoint integral to a Taylor coefficient series."""
    coefficients, one = _adjoint_polynomial_series(angular, xp)
    return taylor_matmul(_adjoint_polynomial_stages(coefficients, generator, xp)[-1], generator, xp) + (
        one * xp.eye(generator.shape[-1], dtype=generator.dtype)
    )


def adjoint_integral_series_jvp(angular, generator, angular_direction, generator_direction, xp):
    coefficients, rates = _adjoint_polynomial_series(angular, xp, derivative=True)
    stages = _adjoint_polynomial_stages(coefficients, generator, xp)
    ds = 2 * taylor_matmul(angular[..., None, :], angular_direction[..., :, None], xp)
    dc = taylor_matmul(ds, rates[..., None, :], xp)[..., 0, :]
    identity = xp.eye(generator.shape[-1], dtype=generator.dtype)
    tangent = dc[..., 3, None, None] * identity
    for stage, i in zip(stages, (2, 1, 0)):
        tangent = (
            taylor_matmul(tangent, generator, xp)
            + taylor_matmul(stage, generator_direction, xp)
            + dc[..., i, None, None] * identity
        )
    return taylor_matmul(tangent, generator, xp) + taylor_matmul(stages[-1], generator_direction, xp)


def adjoint_integral_series_vjp(angular, generator, cotangent, xp):
    coefficients, rates = _adjoint_polynomial_series(angular, xp, derivative=True)
    stages = _adjoint_polynomial_stages(coefficients, generator, xp)
    stage_cotangent, generator_cotangent = taylor_matmul_vjp(stages[-1], generator, cotangent, xp)
    coefficient_cotangents = []
    for stage in reversed(stages[:-1]):
        coefficient_cotangents.append(xp.trace(stage_cotangent, axis1=-2, axis2=-1))
        stage_cotangent, increment = taylor_matmul_vjp(stage, generator, stage_cotangent, xp)
        generator_cotangent = generator_cotangent + increment
    coefficient_cotangents.append(xp.trace(stage_cotangent, axis1=-2, axis2=-1))
    dc = xp.stack(coefficient_cotangents, axis=-1)

    # c(t) = c(s(t)) and s(t) = angular(t) dot angular(t).
    ds = taylor_matmul_left_vjp(rates[..., None, :], dc[..., None, :], xp)
    angular_cotangent = taylor_matmul_left_vjp(angular[..., :, None], ds, xp)
    return 2 * angular_cotangent[..., 0, :], generator_cotangent


def unit_adjoint_integral(vec, directions, angular, generator, xp):
    """Integral over [0, 1] of exp(t K(vec)), or its directional derivatives.

    ``angular(v)`` returns the rotation part whose squared norm is s, and
    ``generator(v)`` the linear map K(v); both are evaluated for ``vec`` and
    for each of the zero, one or two ``directions``.
    """
    inputs = (vec, *directions)
    order = len(directions)
    dtype = xp.result_type(*(value.dtype for value in inputs))
    vec, *directions = (value.astype(dtype, copy=False) for value in inputs)
    rotation = angular(vec)
    theta_squared = xp.sum(rotation * rotation, axis=-1)
    coefficients = adjoint_integral_coefficients(theta_squared, order, xp)
    K = generator(vec)
    identity = xp.eye(K.shape[-1], dtype=dtype)

    if order == 0:
        a1, a2, a3, a4 = coefficients[0]
        polynomial = a4[..., None, None] * K + a3[..., None, None] * identity
        polynomial = polynomial @ K + a2[..., None, None] * identity
        polynomial = polynomial @ K + a1[..., None, None] * identity
        return identity + polynomial @ K

    u = directions[0]
    K_u = generator(u)
    theta_u = 2 * xp.sum(rotation * angular(u), axis=-1)
    if order == 2:
        v = directions[1]
        K_v = generator(v)
        theta_v = 2 * xp.sum(rotation * angular(v), axis=-1)
        theta_uv = 2 * xp.sum(angular(u) * angular(v), axis=-1)

    # Horner evaluation with the product and chain rules at each step.
    c, dc = coefficients[0][3], coefficients[1][3]
    value = c[..., None, None] * identity
    value_u = (dc * theta_u)[..., None, None] * identity
    if order == 2:
        value_v = (dc * theta_v)[..., None, None] * identity
        dc_uv = coefficients[2][3] * theta_u * theta_v + dc * theta_uv
        value_uv = dc_uv[..., None, None] * identity

    for i in range(2, -1, -1):
        c, dc = coefficients[0][i], coefficients[1][i]
        if order == 2:
            dc_uv = coefficients[2][i] * theta_u * theta_v + dc * theta_uv
            value_uv = (value_uv @ K + value_u @ K_v
                        + value_v @ K_u + dc_uv[..., None, None] * identity)
            value_v = (value_v @ K + value @ K_v
                       + (dc * theta_v)[..., None, None] * identity)
        value_u = (value_u @ K + value @ K_u
                   + (dc * theta_u)[..., None, None] * identity)
        value = value @ K + c[..., None, None] * identity
    if order == 2:
        return value_uv @ K + value_u @ K_v + value_v @ K_u
    return value_u @ K + value @ K_u
