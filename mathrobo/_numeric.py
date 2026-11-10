"""Numeric input policy shared by storage and direct Lie operations."""
import numpy as np
import jax.numpy as jnp


def as_inexact(value, LIB=None):
    """Promote integers/bools; preserve floating and complex input precision.

    NumPy integral inputs become float64. JAX uses its configured default float
    (float32 unless x64 is enabled). No conversion of traced values to NumPy.
    """
    if LIB is None:
        LIB = "jax" if type(value).__module__.startswith(("jax", "jaxlib")) else "numpy"
    if LIB not in ("numpy", "jax"):
        raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    xp = jnp if LIB == "jax" else np
    array = xp.asarray(value)
    if array.dtype.kind in "biu":
        return array.astype(xp.asarray(0.0).dtype)
    if array.dtype.kind not in "fc":
        raise TypeError("Expected a numeric array")
    return array


def common_inexact(left, right, LIB):
    """Normalize inputs and use a common dtype without discarding imaginary parts."""
    left, right = as_inexact(left, LIB), as_inexact(right, LIB)
    xp = jnp if LIB == "jax" else np
    dtype = xp.result_type(left.dtype, right.dtype)
    return left.astype(dtype, copy=False), right.astype(dtype, copy=False)
