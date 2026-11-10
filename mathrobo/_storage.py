"""Owned immutable array storage; public objects replace storage on update."""
import numpy as np
from ._numeric import as_inexact


def snapshot(value, LIB=None):
    value = as_inexact(value, LIB)
    if type(value).__module__.startswith(("jax", "jaxlib")):
        return value
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError("Expected a numeric array")
    base = array
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    if isinstance(base, bytes):
        return array.view()
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def same_shape(value, previous, LIB=None):
    value = snapshot(value, LIB)
    if value.shape != previous.shape:
        raise ValueError("Updates must preserve array shape; construct a new object to resize")
    return value


def readonly_view(value):
    """Expose data without sharing mutable NumPy shape metadata."""
    return value.view() if isinstance(value, np.ndarray) else value
