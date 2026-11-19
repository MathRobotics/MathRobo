"""Coefficients of the SE(3) adjoint integral and their angle derivatives."""

from functools import lru_cache
from math import comb, factorial, sin, sqrt

import numpy as np

from .._taylor import taylor_matmul


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


def _adjoint_polynomial_series(angular, xp):
    """Return the four scalar coefficients of the adjoint integral."""
    order = angular.shape[-2]
    angular_row = angular[..., None, :]
    squared = taylor_matmul(angular_row, angular_row.swapaxes(-1, -2), xp)
    base = squared[..., 0, 0, 0]
    derivatives = adjoint_integral_coefficients(
        base, order - 1, xp, normalized=True,
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
    return composition @ derivatives, one


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
