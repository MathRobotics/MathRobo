from .._numeric import as_inexact
from .._storage import snapshot, same_shape, readonly_view, SnapshotState
from typing import Union

import jax
import numpy as np
import jax.numpy as jnp

from .. import _random
from .._batch import array_lib, matvec, transpose_last
from ._se3_jacobian import adjoint_integral_series, unit_adjoint_integral
from .lie_abst import LieAbstract, check_frame

__all__ = ['SO3', 'SO3wrench', 'SO3inertia']


class SO3(SnapshotState, LieAbstract):
    _dof = 3
    _snapshot_fields = ('_rot',)

    def __init__(self, r = np.identity(3), LIB : str = 'numpy'):
        '''
        Constructor
        '''
        self._rot = snapshot(r, LIB)
        self._lib = LIB

    def update_rot(self, rot):
        """Replace rotation, preserving shape and isolating caller storage."""
        self._rot = same_shape(rot, self._rot, self.lib)

    @property
    def lib(self) -> str:
        '''
        Return the library used for the Lie group
        '''
        return self._lib

    @staticmethod
    def dof() -> int:
        return 3
    
    @staticmethod
    def mat_size() -> int:
        return 3
    
    @staticmethod
    def mat_adj_size() -> int:
        return 3
        
    def mat(self) -> Union[np.ndarray, jnp.ndarray]:
        return readonly_view(self._rot)
    
    @classmethod
    def set_mat(cls, mat = np.identity(3), LIB : str = 'numpy') -> 'SO3':
        mat = as_inexact(mat, LIB)
        return cls(mat, LIB)
    
    def quaternion(self) -> Union[np.ndarray, jnp.ndarray]:
        """Unit quaternion ``[..., (w, x, y, z)]`` in this object's backend.

        Branches follow Shepperd's method and are selected with ``where`` so the
        conversion works for batches and under ``jax.jit``.
        """
        xp = array_lib(self.lib)
        r = self._rot
        r00, r11, r22 = r[..., 0, 0], r[..., 1, 1], r[..., 2, 2]
        trace = r00 + r11 + r22
        tiny = xp.finfo(r.dtype).tiny

        def scale(value):
            return 2.0 * xp.sqrt(xp.maximum(value, tiny))

        # Each candidate is accurate where its leading component is large.
        s = scale(1.0 + trace)
        q_w = xp.stack([0.25 * s, (r[..., 2, 1] - r[..., 1, 2]) / s,
                        (r[..., 0, 2] - r[..., 2, 0]) / s, (r[..., 1, 0] - r[..., 0, 1]) / s], axis=-1)
        s = scale(1.0 + r00 - r11 - r22)
        q_x = xp.stack([(r[..., 2, 1] - r[..., 1, 2]) / s, 0.25 * s,
                        (r[..., 0, 1] + r[..., 1, 0]) / s, (r[..., 0, 2] + r[..., 2, 0]) / s], axis=-1)
        s = scale(1.0 + r11 - r00 - r22)
        q_y = xp.stack([(r[..., 0, 2] - r[..., 2, 0]) / s, (r[..., 0, 1] + r[..., 1, 0]) / s,
                        0.25 * s, (r[..., 1, 2] + r[..., 2, 1]) / s], axis=-1)
        s = scale(1.0 + r22 - r00 - r11)
        q_z = xp.stack([(r[..., 1, 0] - r[..., 0, 1]) / s, (r[..., 0, 2] + r[..., 2, 0]) / s,
                        (r[..., 1, 2] + r[..., 2, 1]) / s, 0.25 * s], axis=-1)

        use_w = (trace > 0.0)[..., None]
        use_x = ((r00 > r11) & (r00 > r22))[..., None]
        use_y = (r11 > r22)[..., None]
        q = xp.where(use_w, q_w, xp.where(use_x, q_x, xp.where(use_y, q_y, q_z)))
        return q / xp.linalg.norm(q, axis=-1, keepdims=True)
        
    @staticmethod
    def quaternion_to_mat(quaternion : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        quaternion = as_inexact(quaternion, LIB)
        w, x, y, z = quaternion
        if LIB == 'jax':
            m = jnp.array([
                [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
                [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
                [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
            ])
            return m
        elif LIB == 'numpy':
            m = np.array([
                [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
                [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
                [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
            ])
            return m
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @classmethod
    def set_quaternion(cls, quaternion : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> 'SO3':
        quaternion = as_inexact(quaternion, LIB)
        if quaternion.shape != (4,):
            raise ValueError("Quaternion must be a 4-element vector.")
        return cls(SO3.quaternion_to_mat(quaternion, LIB), LIB)
    
    _EULER_ORDERS = ('ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY')

    @staticmethod
    def _axis_rotation(axis: int, angle, xp):
        """Rotation about coordinate axis 0, 1 or 2, with batch shape angle.shape."""
        c, s = xp.cos(angle), xp.sin(angle)
        zero, one = xp.zeros_like(angle), xp.ones_like(angle)
        if axis == 0:
            rows = ((one, zero, zero), (zero, c, -s), (zero, s, c))
        elif axis == 1:
            rows = ((c, zero, s), (zero, one, zero), (-s, zero, c))
        else:
            rows = ((c, -s, zero), (s, c, zero), (zero, zero, one))
        return xp.stack([xp.stack(row, axis=-1) for row in rows], axis=-2)

    @classmethod
    def set_euler(cls, euler : Union[np.ndarray, jnp.ndarray], order : str = 'ZYX', LIB : str = 'numpy') -> 'SO3':
        """Rotation from angles ``euler[..., :] = (x, y, z)`` about the X, Y and Z axes.

        The matrix is the product of the axis rotations in ``order``; for
        example ``'ZYX'`` gives ``Rz(z) @ Ry(y) @ Rx(x)`` (intrinsic Z-Y-X).
        """
        euler = as_inexact(euler, LIB)
        if euler.ndim == 0 or euler.shape[-1] != 3:
            raise ValueError("Euler angles must have shape (..., 3).")
        if order not in SO3._EULER_ORDERS:
            raise ValueError("Unsupported order. Choose from 'ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY'.")
        xp = array_lib(LIB)
        axes = ['XYZ'.index(name) for name in order]
        m = SO3._axis_rotation(axes[0], euler[..., axes[0]], xp)
        for axis in axes[1:]:
            m = m @ SO3._axis_rotation(axis, euler[..., axis], xp)
        return cls(m, LIB)
    
    @staticmethod
    def mat_to_quaternion(mat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        mat = as_inexact(mat, LIB)
        m = SO3(mat, LIB)
        return m.quaternion()
    
    @classmethod
    def eye(cls, LIB : str = 'numpy') -> 'SO3':
        return cls(array_lib(LIB).identity(3), LIB)

    def inv(self) -> 'SO3':
        return type(self)(transpose_last(self._rot), self.lib)

    def mat_inv(self) -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(self._rot)

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        return readonly_view(self._rot)
    
    @staticmethod
    def set_mat_adj(mat = np.identity(3), LIB : str = 'numpy') -> 'SO3':
        mat = as_inexact(mat, LIB)
        return SO3(mat, LIB)

    def mat_inv_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(self._rot)

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
        mat = xp.zeros(vec.shape[:-1] + (3, 3), dtype=vec.dtype)
        if LIB == "jax":
            mat = mat.at[..., 0, 1].set(-vz)
            mat = mat.at[..., 0, 2].set(vy)
            mat = mat.at[..., 1, 0].set(vz)
            mat = mat.at[..., 1, 2].set(-vx)
            mat = mat.at[..., 2, 0].set(-vy)
            mat = mat.at[..., 2, 1].set(vx)
            return mat
        elif LIB == "numpy":
            mat[..., 0, 1] = -vz
            mat[..., 0, 2] = vy
            mat[..., 1, 0] = vz
            mat[..., 1, 2] = -vx
            mat[..., 2, 0] = -vy
            mat[..., 2, 1] = vx
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.hat(vec, LIB)

    @staticmethod
    def vee(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec_hat = as_inexact(vec_hat, LIB)
        if vec_hat.shape[-2:] != (3, 3):
            raise ValueError("Input matrix must be of size (...,3,3).")
        if LIB == 'jax':
            return jnp.stack([
                0.5 * (vec_hat[..., 2, 1] - vec_hat[..., 1, 2]),
                0.5 * (vec_hat[..., 0, 2] - vec_hat[..., 2, 0]),
                0.5 * (vec_hat[..., 1, 0] - vec_hat[..., 0, 1])
            ], axis=-1)
        elif LIB == 'numpy':
            return np.stack([
                0.5 * (vec_hat[..., 2, 1] - vec_hat[..., 1, 2]),
                0.5 * (vec_hat[..., 0, 2] - vec_hat[..., 2, 0]),
                0.5 * (vec_hat[..., 1, 0] - vec_hat[..., 0, 1])
            ], axis=-1).astype(vec_hat.dtype, copy=False)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        theta2 = xp.sum(vec * vec, axis=-1)
        x2 = a * a * theta2
        series = xp.abs(x2) < 1e-6
        # Keep the inactive closed-form branch differentiable at zero as well.
        closed_theta2 = xp.where(series, xp.ones_like(theta2), theta2)
        closed_theta = xp.sqrt(closed_theta2)
        closed_angle = a * closed_theta
        A_closed = xp.sin(closed_angle) / closed_theta
        B_closed = (1.0 - xp.cos(closed_angle)) / closed_theta2
        A_series = a * (1 - x2/6 + x2*x2/120 - x2*x2*x2/5040)
        B_series = a*a * (1/2 - x2/24 + x2*x2/720 - x2*x2*x2/40320)
        A = xp.where(series, A_series, A_closed)
        B = xp.where(series, B_series, B_closed)
        K = SO3.hat(vec, LIB)
        K2 = K @ K
        I = xp.eye(3, dtype=vec.dtype)
        return I + A[..., None, None] * K + B[..., None, None] * K2
    
    @staticmethod
    def _log_mat(rot, LIB):
        """Principal rotation vector of a real rotation matrix."""
        xp = array_lib(LIB)
        skew = SO3.vee(rot, LIB)
        sin_squared = xp.sum(skew * skew, axis=-1)
        cosine = (xp.trace(rot, axis1=-2, axis2=-1) - 1) / 2
        near_identity = (cosine > 0) & (sin_squared < 1e-6)
        near_pi = cosine < -0.5

        sine = xp.sqrt(xp.where(near_identity, xp.ones_like(sin_squared), sin_squared))
        angle = xp.arctan2(sine, cosine)
        scale = angle / xp.where(near_pi, xp.ones_like(sine), sine)
        series = 1 + sin_squared * (1 / 6 + sin_squared * (3 / 40))
        ordinary = xp.where(near_identity, series, scale)[..., None] * skew

        # The symmetric part gives (1 - cos(angle)) times the axis outer product.
        # Its largest diagonal selects a well-scaled row even at angle pi.
        identity = xp.eye(3, dtype=rot.dtype)
        symmetric = (rot + transpose_last(rot)) / 2 - cosine[..., None, None] * identity
        index = xp.argmax(xp.diagonal(symmetric, axis1=-2, axis2=-1), axis=-1)
        axis = xp.take_along_axis(symmetric, index[..., None, None], axis=-2)[..., 0, :]
        axis = xp.where(near_pi[..., None], axis, identity[0])
        axis = axis / xp.sqrt(xp.sum(axis * axis, axis=-1, keepdims=True))
        sign = xp.where(xp.sum(axis * skew, axis=-1) < 0, -1, 1).astype(rot.dtype)
        return xp.where(near_pi[..., None], (sign * angle)[..., None] * axis, ordinary)

    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        theta2 = xp.sum(vec * vec, axis=-1)
        x2 = a * a * theta2
        series = xp.abs(x2) < 1e-6
        closed_theta2 = xp.where(series, xp.ones_like(theta2), theta2)
        closed_theta = xp.sqrt(closed_theta2)
        closed_angle = a * closed_theta
        K = SO3.hat(vec, LIB)
        K2 = K @ K
        # The half-angle form avoids subtracting nearly equal values in float32.
        A_closed = 2 * (xp.sin(closed_angle / 2) / closed_theta)**2
        B_closed = (closed_angle - xp.sin(closed_angle)) / (closed_theta2 * closed_theta)
        A_series = a*a * (1/2 - x2/24 + x2*x2/720 - x2*x2*x2/40320)
        B_series = a*a*a * (1/6 - x2/120 + x2*x2/5040 - x2*x2*x2/362880)
        A = xp.where(series, A_series, A_closed)
        B = xp.where(series, B_series, B_closed)
        I = xp.eye(3, dtype=vec.dtype)
        result = a * I + A[..., None, None] * K + B[..., None, None] * K2
        return result
    
    @staticmethod
    def exp_integ2nd(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        theta2 = xp.sum(vec * vec, axis=-1)
        x2 = a * a * theta2
        series = xp.abs(x2) < 1e-6
        closed_theta2 = xp.where(series, xp.ones_like(theta2), theta2)
        closed_theta = xp.sqrt(closed_theta2)
        closed_angle = a * closed_theta
        K = SO3.hat(vec, LIB)
        K2 = K @ K
        A_closed = (closed_angle - xp.sin(closed_angle)) / (closed_theta2 * closed_theta)
        B_closed = (0.5 * closed_angle * closed_angle - 1.0 + xp.cos(closed_angle)) / (closed_theta2 * closed_theta2)
        A_series = a*a*a * (1/6 - x2/120 + x2*x2/5040 - x2*x2*x2/362880)
        B_series = a**4 * (1/24 - x2/720 + x2*x2/40320 - x2*x2*x2/3628800)
        A = xp.where(series, A_series, A_closed)
        B = xp.where(series, B_series, B_closed)
        I = xp.eye(3, dtype=vec.dtype)
        result = 0.5 * a * a * I + A[..., None, None] * K + B[..., None, None] * K2
        return result
    
    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.hat(vec, LIB)
    
    @staticmethod
    def hat_commute_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.hat_commute(vec, LIB)
    
    @staticmethod
    def vee_adj(mat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.vee(mat, LIB)
    
    @staticmethod
    def exp_adj(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.exp(vec, a, LIB)
    
    @staticmethod
    def exp_integ_adj(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.exp_integ(vec, a, LIB)

    @classmethod
    def right_jacobian(cls, phi, LIB: str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Return Jr(phi) with ``Exp(phi + d) ~= Exp(phi) Exp(Jr(phi) d)``, shape (..., 3, 3)."""
        return cls._unit_integral(-as_inexact(phi, LIB), (), LIB)

    @classmethod
    def right_jacobian_directional(cls, phi, u, LIB: str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Return DJ_r(phi)[u] for a direction in the same rotation coordinates."""
        return cls._unit_integral(-as_inexact(phi, LIB), (-as_inexact(u, LIB),), LIB)

    @classmethod
    def right_jacobian_second_directional(cls, phi, u, v, LIB: str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Return the symmetric mixed directional derivative D²J_r(phi)[u, v]."""
        directions = (-as_inexact(u, LIB), -as_inexact(v, LIB))
        return cls._unit_integral(-as_inexact(phi, LIB), directions, LIB)

    @classmethod
    def _unit_integral(cls, vec, directions, LIB):
        """Integral over [0, 1] of exp(t hat(vec)) with 3 x 3 generators."""
        if any(value.ndim == 0 or value.shape[-1] != 3 for value in (vec, *directions)):
            raise ValueError("Input vector and directions must have shape (..., 3)")
        return unit_adjoint_integral(vec, directions, lambda v: v,
                                     lambda v: cls.hat(v, LIB), array_lib(LIB))

    @classmethod
    def _right_jacobian_series(cls, coordinates, LIB):
        if coordinates.shape[-2] == 1:
            return cls.right_jacobian(coordinates[..., 0, :], LIB)[..., None, :, :]
        return adjoint_integral_series(coordinates, cls.hat(-coordinates, LIB), array_lib(LIB))

    @staticmethod
    def sub_tan_vec(val0 : 'SO3', val1 : 'SO3', 
                    frame : str = 'bframe', LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        check_frame(frame)
        if frame == 'bframe':
            vec = type(val0).vee(val0.mat_inv() @ (val1._rot - val0._rot), LIB)
        elif frame == 'fframe':
            vec = type(val0).vee((val1._rot - val0._rot) @ val0.mat_inv(), LIB)
        return vec
    
    @staticmethod
    def so3_mul(l_rot : Union[np.ndarray, jnp.ndarray], r_rot : Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        assert isinstance(l_rot, jnp.ndarray) or isinstance(l_rot, np.ndarray), "Input must be a numpy or jax array."
        assert isinstance(r_rot, jnp.ndarray) or isinstance(r_rot, np.ndarray), "Input must be a numpy or jax array."
        return l_rot @ r_rot

    def __matmul__(self, rval):
        if isinstance(rval, SO3):
            return type(self)(SO3.so3_mul(self._rot, rval._rot), self.lib)
        elif isinstance(rval, (np.ndarray, jnp.ndarray)):
            if rval.shape[-1] == 3 and (rval.ndim == 1 or rval.shape[-2:] != (3, 3)):
                return matvec(self._rot, rval)
            return SO3.so3_mul(self._rot, rval)
        else:
            raise TypeError("Right operand should be SO3, numpy.ndarray, or jax.ndarray")

    @classmethod
    def rand(cls, LIB : str = 'numpy', key=None) -> 'SO3':
        """Uniformly distributed (Haar) rotation; see ``mathrobo._random`` for ``key``."""
        array_lib(LIB)  # reject unsupported backends early
        q = _random.normal(key, (4,), LIB)
        q = q / array_lib(LIB).linalg.norm(q)
        return cls(SO3.quaternion_to_mat(q, LIB), LIB)

    def __repr__(self):
        return f"SO3(\nrot=\n{self._rot},\nLIB='{self.lib}')"
    
class SO3wrench(SO3):
    """SO3 with negated algebra coordinates: ``hat(w) = -SO3.hat(w)``.

    ``vee``, ``hat_adj`` and the exponentials use the same coordinates, so
    ``vee(hat(w)) == w`` and ``hat_adj(x) @ y == vee([hat(x), hat(y)])``.
    """
    @staticmethod
    def vee(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.vee(vec_hat, LIB)

    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.hat(vec, LIB)

    @staticmethod
    def hat_commute_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.hat(vec, LIB)

    @staticmethod
    def vee_adj(mat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.vee(mat, LIB)

    @staticmethod
    def exp_adj(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3wrench.exp(vec, a, LIB)

    @staticmethod
    def exp_integ_adj(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3wrench.exp_integ(vec, a, LIB)

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.hat(vec, LIB)
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.hat(vec, LIB)

    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SO3.exp(vec, a, LIB))
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SO3.exp_integ(vec, a, LIB))
    
class SO3inertia(SO3):
    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if LIB == 'jax':
            return jnp.array([
                [vec[0], vec[5], vec[4]],
                [vec[5], vec[1], vec[3]],
                [vec[4], vec[3], vec[2]],
            ], dtype=vec.dtype)
        elif LIB == 'numpy':
            return np.array([
                [vec[0], vec[5], vec[4]],
                [vec[5], vec[1], vec[3]],
                [vec[4], vec[3], vec[2]],
            ], dtype=vec.dtype)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if LIB == 'jax':
            return jnp.array([
                [vec[0],      0,      0,      0, vec[2], vec[1]],
                [     0, vec[1],      0, vec[2],      0, vec[0]],
                [     0,      0, vec[2], vec[1], vec[0],      0],
            ], dtype=vec.dtype)
        elif LIB == 'numpy':
            return np.array([
                [vec[0],      0,      0,      0, vec[2], vec[1]],
                [     0, vec[1],      0, vec[2],      0, vec[0]],
                [     0,      0, vec[2], vec[1], vec[0],      0],
            ], dtype=vec.dtype)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SO3.exp(vec, a, LIB))
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SO3.exp_integ(vec, a, LIB))
