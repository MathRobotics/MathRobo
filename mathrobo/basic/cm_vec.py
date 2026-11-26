from .._storage import snapshot, same_shape, readonly_view, SnapshotState
import math
import operator
from typing import Union

import numpy as np
import jax.numpy as jnp

from .._batch import flatten_last2
from .._numeric import as_inexact


__all__ = ['CMVector']


# Shared coefficients through the tenth derivative (orders 0 through 10).
_FACTORIALS = np.asarray([math.factorial(i) for i in range(11)], dtype=np.float64)
_INVERSE_FACTORIALS = 1.0 / _FACTORIALS
_FACTORIALS.setflags(write=False)
_INVERSE_FACTORIALS.setflags(write=False)


class CMVector(SnapshotState):
    """A derivative series stored as CM coefficients, ``vecs[k] / k!``.

    ``CMVector(vecs)`` accepts ordinary derivatives. ``set_cmvecs`` accepts
    normalized coefficients directly, capturing an independent snapshot.
    ``vecs()`` and ``vec()`` reconstruct ordinary derivatives on each call.
    These reconstructed arrays do not share storage with the coefficients.
    Integer inputs are promoted to floating point; floating and complex
    precision is preserved. Ordinary outputs may differ by rounding error.
    Arrays have shape ``(..., order, dimension)``; flattening preserves batches.
    """

    _snapshot_fields = ('_cm_vecs',)

    def __init__(self, vecs: Union[np.ndarray, jnp.ndarray]):
        vecs = as_inexact(vecs)
        self._set_cm_vecs(vecs, squeeze_vec=vecs.ndim == 1)
        self._cm_vecs = snapshot(self._cm_vecs * self._scales(inverse=True))

    def _set_cm_vecs(self, cm_vecs, squeeze_vec=False):
        cm_vecs = as_inexact(cm_vecs)
        self._squeeze_vec = squeeze_vec
        if cm_vecs.ndim == 1:
            cm_vecs = cm_vecs[..., None]
        self._cm_vecs = snapshot(cm_vecs)
        self._n, self._dim = cm_vecs.shape[-2:]
        self._len = self._n * self._dim

    def _scales(self, inverse=False):
        if self._n <= len(_FACTORIALS):
            table = _INVERSE_FACTORIALS if inverse else _FACTORIALS
            scales = table[:self._n]
        else:
            scales = np.asarray([math.factorial(i) for i in range(self._n)], dtype=np.float64)
            if inverse:
                scales = 1.0 / scales

        module = type(self._cm_vecs).__module__
        if module.startswith(("jax", "jaxlib")):
            dtype = self._cm_vecs.dtype
            scales = jnp.asarray(scales, dtype=dtype)
        else:
            dtype = self._cm_vecs.dtype
            scales = scales.astype(dtype, copy=False)
        return scales[:, None]

    @classmethod
    def set_cmvecs(cls, cm_vecs: Union[np.ndarray, jnp.ndarray]) -> 'CMVector':
        obj = cls.__new__(cls)
        # Preserve the existing column-series layout for one-dimensional input.
        obj._set_cm_vecs(cm_vecs)
        return obj

    def update_vecs(self, vecs):
        """Replace ordinary derivatives without changing shape or layout."""
        replacement = CMVector(vecs)
        if replacement._squeeze_vec != self._squeeze_vec:
            raise ValueError("Updates must preserve series layout")
        self._cm_vecs = same_shape(replacement._cm_vecs, self._cm_vecs)

    def update_cmvecs(self, coefficients):
        """Replace normalized coefficients in the cm_vecs() layout."""
        values = same_shape(coefficients, self.cm_vecs())
        self._cm_vecs = values[..., None] if self._squeeze_vec else values

    def vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        vecs = self._cm_vecs * self._scales()
        return vecs[..., 0] if self._squeeze_vec else vecs

    def cm_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return readonly_view(self._cm_vecs[..., 0] if self._squeeze_vec else self._cm_vecs)

    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        vecs = self.vecs()
        return vecs if self._squeeze_vec else flatten_last2(vecs)

    def cm_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return flatten_last2(self._cm_vecs)

    def _cm_result(self, coefficients, *, squeeze_vec):
        result = type(self).set_cmvecs(coefficients)
        result._squeeze_vec = squeeze_vec
        return result

    def truncate(self, order: int) -> 'CMVector':
        """Return the first order coefficient rows, sharing array storage.

        order is a row count (0 through self._n), not a derivative index.
        One-dimensional series keep their original output layout.
        """
        if isinstance(order, (bool, np.bool_)):
            raise TypeError("order must be an integer row count")
        order = operator.index(order)
        if not 0 <= order <= self._n:
            raise ValueError(f"order must be between 0 and {self._n}, got {order}")
        return self._cm_result(
            self._cm_vecs[..., :order, :], squeeze_vec=self._squeeze_vec
        )

    def _combine(self, other, operation):
        if not isinstance(other, CMVector):
            return NotImplemented
        if self._n != other._n or self._dim != other._dim:
            raise ValueError("CMVector orders and dimensions must match")
        # Only batch axes broadcast; order and dimension axes must match exactly.
        coefficients = operation(self._cm_vecs, other._cm_vecs)
        return self._cm_result(
            coefficients, squeeze_vec=self._squeeze_vec and other._squeeze_vec
        )

    def __add__(self, other):
        """Add matching series, broadcasting their batch axes."""
        return self._combine(other, operator.add)

    def __sub__(self, other):
        """Subtract matching series, broadcasting their batch axes."""
        return self._combine(other, operator.sub)

    def __neg__(self):
        """Negate coefficients without reconstructing ordinary derivatives."""
        return self._cm_result(-self._cm_vecs, squeeze_vec=self._squeeze_vec)

    def __repr__(self):
        return f"CMVector(n={self._n}, dim={self._dim}, len={self._len})\n{self.cm_vecs()}"
