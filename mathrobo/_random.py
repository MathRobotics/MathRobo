"""Random draws shared by ``rand`` constructors on both backends.

``key`` may be None, an integer seed, a ``numpy.random.Generator`` (NumPy) or
a JAX PRNG key (JAX). Without a key, NumPy uses its global RNG and JAX derives
a fresh key from it, so ``np.random.seed`` makes both reproducible.
"""
import numpy as np
import jax


def _numpy_rng(key):
    if key is None:
        return np.random
    if isinstance(key, np.random.Generator):
        return key
    if isinstance(key, (int, np.integer)):
        return np.random.default_rng(int(key))
    raise TypeError("NumPy key must be None, an integer or numpy.random.Generator")


def _jax_key(key):
    if key is None:
        return jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    return key


def split(key, LIB, count):
    """Independent keys for ``count`` draws from one user-supplied key."""
    if LIB == "jax":
        return tuple(jax.random.split(_jax_key(key), count))
    # One generator is shared so successive draws stay independent.
    rng = None if key is None else _numpy_rng(key)
    return (rng,) * count


def normal(key, shape, LIB):
    if LIB == "jax":
        return jax.random.normal(_jax_key(key), shape)
    return _numpy_rng(key).standard_normal(shape)


def uniform(key, shape, LIB, low=0.0, high=1.0):
    if LIB == "jax":
        return jax.random.uniform(_jax_key(key), shape, minval=low, maxval=high)
    return _numpy_rng(key).uniform(low, high, shape)
