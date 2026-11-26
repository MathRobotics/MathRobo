"""Planar rotations: two-dimensional matrices and a one-dimensional Lie algebra."""
import warnings
import numpy as np
import jax
import jax.numpy as jnp

from .. import _random
from .._batch import array_lib, matvec, transpose_last
from .._numeric import as_inexact
from .._storage import SnapshotState, snapshot, same_shape, readonly_view
from .lie_abst import LieAbstract


__all__ = ['SO2']


class SO2(SnapshotState, LieAbstract):
    _dof = 1
    _snapshot_fields = ('_rot',)

    def __init__(self, r=np.identity(2), LIB='numpy'):
        r = snapshot(r, LIB)
        if r.ndim < 2 or r.shape[-2:] != (2, 2):
            raise ValueError('Rotation must have shape (..., 2, 2)')
        self._rot, self._lib = r, LIB

    @property
    def lib(self):
        return self._lib

    @staticmethod
    def dof():
        return 1

    @staticmethod
    def mat_size():
        return 2

    @staticmethod
    def mat_adj_size():
        return 1

    def mat(self):
        return readonly_view(self._rot)

    def update_rot(self, rot):
        self._rot = same_shape(rot, self._rot, self.lib)

    @classmethod
    def set_mat(cls, mat=np.identity(2), LIB='numpy'):
        return cls(mat, LIB)

    def angle(self):
        """Principal angle in [-pi, pi], preserving batch axes."""
        return array_lib(self.lib).arctan2(self._rot[..., 1, 0], self._rot[..., 0, 0])

    @staticmethod
    def angle_to_rot_mat(angle, LIB='numpy'):
        angle = as_inexact(angle, LIB)
        xp = array_lib(LIB)
        c, s = xp.cos(angle), xp.sin(angle)
        return xp.stack((xp.stack((c, -s), axis=-1), xp.stack((s, c), axis=-1)), axis=-2)

    @classmethod
    def set_angle(cls, angle, LIB='numpy'):
        return cls(cls.angle_to_rot_mat(angle, LIB), LIB)

    @classmethod
    def set_quaternion(cls, quaternion, LIB='numpy'):
        """Deprecated legacy alias accepting an angle, not a quaternion."""
        warnings.warn('SO2.set_quaternion accepts an angle; use set_angle instead',
                      DeprecationWarning, stacklevel=2)
        return cls.set_angle(quaternion, LIB)

    @classmethod
    def eye(cls, LIB='numpy'):
        return cls(array_lib(LIB).eye(2), LIB)

    @classmethod
    def rand(cls, LIB='numpy', key=None):
        """Uniform angle in [-pi, pi); see ``mathrobo._random`` for ``key``."""
        array_lib(LIB)  # reject unsupported backends early
        return cls.set_angle(_random.uniform(key, (), LIB, -np.pi, np.pi), LIB)

    def inv(self):
        return type(self)(self.mat_inv(), self.lib)

    def mat_inv(self):
        return transpose_last(self._rot)

    def mat_adj(self):
        return array_lib(self.lib).ones(self._rot.shape[:-2] + (1, 1), dtype=self._rot.dtype)

    def mat_inv_adj(self):
        return self.mat_adj()

    def inv_adj(self):
        """Compatibility alias for mat_inv_adj."""
        return self.mat_inv_adj()

    @staticmethod
    def set_mat_adj(mat=None, LIB='numpy'):
        raise ValueError('SO2 adjoint is always identity and cannot determine a rotation; use set_angle or set_mat')

    @staticmethod
    def _vector(vec, LIB):
        vec = as_inexact(vec, LIB)
        if vec.ndim == 0:
            vec = vec.reshape(1)
        if vec.shape[-1] != 1:
            raise ValueError('SO2 tangent vectors must have shape (..., 1)')
        return vec

    @staticmethod
    def hat(vec, LIB='numpy'):
        vec = SO2._vector(vec, LIB)
        return vec[..., 0, None, None] * array_lib(LIB).asarray([[0, -1], [1, 0]], dtype=vec.dtype)

    @staticmethod
    def vee(matrix, LIB='numpy'):
        matrix = as_inexact(matrix, LIB)
        if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
            raise ValueError('Expected shape (..., 2, 2)')
        return ((matrix[..., 1, 0] - matrix[..., 0, 1]) * .5)[..., None]

    @staticmethod
    def hat_commute(vec, LIB='numpy'):
        vec = as_inexact(vec, LIB)
        if vec.ndim == 0 or vec.shape[-1] != 2:
            raise ValueError('Expected spatial vector shape (..., 2)')
        return array_lib(LIB).stack((-vec[..., 1], vec[..., 0]), axis=-1)[..., None]

    @staticmethod
    def hat_adj(vec, LIB='numpy'):
        vec = SO2._vector(vec, LIB)
        return array_lib(LIB).zeros(vec.shape[:-1] + (1, 1), dtype=vec.dtype)

    hat_commute_adj = hat_adj

    @staticmethod
    def vee_adj(matrix, LIB='numpy'):
        raise ValueError('SO2 adjoint Lie algebra is zero and cannot recover angular velocity; use vee')

    @staticmethod
    def exp(vec, a=1., LIB='numpy'):
        return SO2.angle_to_rot_mat(a * SO2._vector(vec, LIB)[..., 0], LIB)

    @staticmethod
    def _coefficients(x, LIB):
        xp = array_lib(LIB)
        small = xp.abs(x) < 1e-3
        safe = xp.where(small, xp.ones_like(x), x)
        x2 = x*x
        sinc = xp.where(small, 1-x2/6+x2*x2/120-x2**3/5040, xp.sin(safe)/safe)
        cosc = xp.where(small, .5-x2/24+x2*x2/720-x2**3/40320, (1-xp.cos(safe))/safe**2)
        third = xp.where(small, 1/6-x2/120+x2*x2/5040-x2**3/362880, (safe-xp.sin(safe))/safe**3)
        return sinc, cosc, third

    @staticmethod
    def exp_integ(vec, a=1., LIB='numpy'):
        vec = SO2._vector(vec, LIB)
        x = a * vec[..., 0]
        sinc, cosc, _ = SO2._coefficients(x, LIB)
        xp = array_lib(LIB)
        return a*sinc[..., None, None]*xp.eye(2, dtype=vec.dtype) + a*a*cosc[..., None, None]*SO2.hat(vec, LIB)

    @staticmethod
    def exp_integ2nd(vec, a=1., LIB='numpy'):
        vec = SO2._vector(vec, LIB)
        x = a * vec[..., 0]
        _, cosc, third = SO2._coefficients(x, LIB)
        xp = array_lib(LIB)
        return a*a*cosc[..., None, None]*xp.eye(2, dtype=vec.dtype) + a**3*third[..., None, None]*SO2.hat(vec, LIB)

    @staticmethod
    def exp_adj(vec, a=1., LIB='numpy'):
        vec = SO2._vector(vec, LIB)
        return array_lib(LIB).ones(vec.shape[:-1] + (1, 1), dtype=vec.dtype)

    @staticmethod
    def exp_integ_adj(vec, a=1., LIB='numpy'):
        return a * SO2.exp_adj(vec, a, LIB)

    @staticmethod
    def sub_tan_vec(val0, val1, frame='bframe', LIB='numpy'):
        if frame not in ('bframe', 'fframe'):
            raise ValueError('Frame must be bframe or fframe')
        return SO2.vee(val0.mat_inv() @ (val1.mat()-val0.mat()), LIB)

    def __matmul__(self, other):
        if isinstance(other, SO2):
            if self.lib != other.lib:
                raise ValueError('SO2 backends must match')
            return type(self)(self._rot @ other._rot, self.lib)
        if isinstance(other, (np.ndarray, jnp.ndarray)):
            if other.shape[-1:] == (2,) and (other.ndim == 1 or other.shape[-2:] != (2, 2)):
                return matvec(self._rot, other)
            return self._rot @ other
        raise TypeError('Right operand should be SO2, numpy.ndarray, or jax.ndarray')

    def __repr__(self):
        return f"SO2(rot={self._rot!r}, LIB={self.lib!r})"
