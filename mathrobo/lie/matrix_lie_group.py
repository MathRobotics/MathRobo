"""Public extension base for user-defined matrix Lie groups (no registration)."""
from abc import abstractmethod
import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from scipy.linalg import expm

from .._batch import array_lib, matvec
from .._numeric import as_inexact
from .._storage import SnapshotState, snapshot, same_shape, readonly_view
from .lie_abst import LieAbstract

__all__ = ['MatrixLieGroup']


class MatrixLieGroup(SnapshotState, LieAbstract):
    """Implement dof(), mat_size(), hat(vec, LIB), and vee(matrix, LIB).

    hat/vee must be linear inverse maps on the algebra, supporting trailing
    dimensions (..., dof) and (..., size, size). The defaults implement matrix
    group operations, adjoints, exponentials and integrals. Subclasses own the
    definition and validation of their group; shape validation alone cannot
    prove membership. Defaults favor correctness over specialized performance.
    """
    _snapshot_fields = ('_matrix',)

    @staticmethod
    @abstractmethod
    def dof():
        """Dimension of the Lie algebra (positive integer)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def mat_size():
        """Size of the square matrix representation (positive integer)."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def hat(vec, LIB='numpy'):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vee(matrix, LIB='numpy'):
        raise NotImplementedError

    def __init__(self, matrix, LIB='numpy'):
        matrix = snapshot(matrix, LIB)
        if matrix.ndim < 2 or matrix.shape[-2:] != (self.mat_size(), self.mat_size()):
            raise ValueError('Matrix shape must end in (mat_size(), mat_size())')
        if self.dof() < 1 or self.mat_size() < 1:
            raise ValueError('dof and mat_size must be positive')
        self._matrix, self._lib = matrix, LIB

    @property
    def lib(self):
        return self._lib

    @classmethod
    def mat_adj_size(cls):
        return cls.dof()

    def mat(self):
        return readonly_view(self._matrix)

    def update_mat(self, matrix):
        """Replace owned matrix storage, preserving batch shape."""
        self._matrix = same_shape(matrix, self._matrix, self.lib)

    @classmethod
    def set_mat(cls, matrix, LIB='numpy'):
        return cls(matrix, LIB)

    @classmethod
    def eye(cls, LIB='numpy'):
        return cls(array_lib(LIB).eye(cls.mat_size()), LIB)

    def mat_inv(self):
        return array_lib(self.lib).linalg.inv(self._matrix)

    def inv(self):
        return type(self).set_mat(self.mat_inv(), LIB=self.lib)

    @classmethod
    def _basis(cls, dtype, LIB):
        return array_lib(LIB).eye(cls.dof(), dtype=dtype)

    def mat_adj(self):
        cls, xp = type(self), array_lib(self.lib)
        inverse = self.mat_inv()
        return xp.stack([cls.vee(self._matrix @ cls.hat(e, self.lib) @ inverse, self.lib)
                         for e in cls._basis(self._matrix.dtype, self.lib)], axis=-1)

    def mat_inv_adj(self):
        return self.inv().mat_adj()

    @classmethod
    def hat_adj(cls, vec, LIB='numpy'):
        vec = as_inexact(vec, LIB)
        h = cls.hat(vec, LIB)
        return array_lib(LIB).stack([
            cls.vee(h @ cls.hat(e, LIB) - cls.hat(e, LIB) @ h, LIB)
            for e in cls._basis(vec.dtype, LIB)], axis=-1)

    @classmethod
    def hat_commute(cls, vec, LIB='numpy'):
        vec = as_inexact(vec, LIB)
        return array_lib(LIB).stack([matvec(cls.hat(e, LIB), vec)
                                    for e in cls._basis(vec.dtype, LIB)], axis=-1)

    @classmethod
    def hat_commute_adj(cls, vec, LIB='numpy'):
        vec = as_inexact(vec, LIB)
        return array_lib(LIB).stack([matvec(cls.hat_adj(e, LIB), vec)
                                    for e in cls._basis(vec.dtype, LIB)], axis=-1)

    @classmethod
    def set_mat_adj(cls, matrix, LIB='numpy'):
        raise NotImplementedError('Adjoint recovery is not unique in general; override it only when mathematically valid')

    @classmethod
    def vee_adj(cls, matrix, LIB='numpy'):
        raise NotImplementedError('Adjoint algebra recovery is not injective in general; override it when valid')

    @staticmethod
    def _expm(matrix, LIB):
        xp = array_lib(LIB)
        size = matrix.shape[-1]
        if matrix.ndim == 2:
            return jsl.expm(matrix) if LIB == 'jax' else expm(matrix)
        flat = matrix.reshape((-1, size, size))
        if flat.shape[0] == 0:
            return xp.empty_like(matrix)
        result = jax.vmap(jsl.expm)(flat) if LIB == 'jax' else np.stack([expm(m) for m in flat])
        return result.reshape(matrix.shape)

    @classmethod
    def exp(cls, vec, a=1., LIB='numpy'):
        return cls._expm(a * cls.hat(as_inexact(vec, LIB), LIB), LIB)

    @classmethod
    def exp_adj(cls, vec, a=1., LIB='numpy'):
        return cls._expm(a * cls.hat_adj(vec, LIB), LIB)

    @classmethod
    def _integral(cls, generator, a, order, LIB):
        xp = array_lib(LIB)
        size = generator.shape[-1]
        eye = xp.broadcast_to(xp.eye(size, dtype=generator.dtype), generator.shape)
        zero = xp.zeros_like(generator)
        rows = []
        for i in range(order+1):
            rows.append(xp.concatenate([generator if i == j == 0 else eye if j == i+1 else zero
                                        for j in range(order+1)], axis=-1))
        matrix = xp.concatenate(rows, axis=-2)
        return cls._expm(a * matrix, LIB)[..., :size, -size:]

    @classmethod
    def exp_integ(cls, vec, a=1., LIB='numpy'):
        return cls._integral(cls.hat(as_inexact(vec, LIB), LIB), a, 1, LIB)

    @classmethod
    def exp_integ2nd(cls, vec, a=1., LIB='numpy'):
        return cls._integral(cls.hat(as_inexact(vec, LIB), LIB), a, 2, LIB)

    @classmethod
    def exp_integ_adj(cls, vec, a=1., LIB='numpy'):
        return cls._integral(cls.hat_adj(vec, LIB), a, 1, LIB)

    def __matmul__(self, other):
        if isinstance(other, MatrixLieGroup):
            if type(self) is not type(other) or self.lib != other.lib:
                raise TypeError('Group type and backend must match')
            return type(self).set_mat(self._matrix @ other._matrix, LIB=self.lib)
        if isinstance(other, (np.ndarray, jnp.ndarray)):
            if other.ndim == 1:
                return matvec(self._matrix, other)
            return self._matrix @ other
        return NotImplemented
