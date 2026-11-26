from .._numeric import as_inexact, common_inexact
from .._storage import snapshot, same_shape, readonly_view, SnapshotState
from typing import Union, Tuple
import math

import jax
import numpy as np
import jax.numpy as jnp

from .._batch import array_lib, matvec, transpose_last
from ._se3_jacobian import adjoint_integral_coefficients, adjoint_integral_series
from .lie_abst import LieAbstract, check_frame
from .so3 import SO3, SO3inertia, SO3wrench

class SE3(SnapshotState, LieAbstract):
    _dof = 6
    _cls_so3 = SO3
    _snapshot_fields = ('_rot', '_pos')

    def __init__(self, rot = np.identity(3), pos = np.zeros(3), LIB : str = 'numpy'): 
        '''
        Constructor
        '''
        self._rot, self._pos = self._pose_storage(rot, pos, LIB)
        self._lib = LIB

    @staticmethod
    def _pose_storage(rot, pos, LIB):
        rot, pos = common_inexact(rot, pos, LIB)
        if rot.ndim < 2 or rot.shape[-2:] != (3, 3):
            raise ValueError("Rotation must have shape (..., 3, 3)")
        if pos.ndim < 1 or pos.shape[-1] != 3:
            raise ValueError("Position must have shape (..., 3)")
        try:
            batch_shape = np.broadcast_shapes(rot.shape[:-2], pos.shape[:-1])
        except ValueError as exc:
            raise ValueError("Rotation and position batch shapes must broadcast") from exc
        xp = array_lib(LIB)
        rot = xp.broadcast_to(rot, batch_shape + (3, 3))
        pos = xp.broadcast_to(pos, batch_shape + (3,))
        return snapshot(rot), snapshot(pos)

    def update_pos(self, pos):
        """Replace translation, promoting rotation and position to a common dtype."""
        pos = same_shape(pos, self._pos, self.lib)
        self._rot, self._pos = self._pose_storage(self._rot, pos, self.lib)

    def update_rot(self, rot):
        """Replace rotation, promoting rotation and position to a common dtype."""
        rot = same_shape(rot, self._rot, self.lib)
        self._rot, self._pos = self._pose_storage(rot, self._pos, self.lib)

    def update_pose(self, rot, pos):
        """Atomically replace rotation and translation, preserving shapes."""
        rotation = same_shape(rot, self._rot, self.lib)
        position = same_shape(pos, self._pos, self.lib)
        self._rot, self._pos = self._pose_storage(rotation, position, self.lib)

    @property
    def lib(self) -> str:
        '''
        Return the library used for the Lie group
        '''
        return self._lib

    @staticmethod
    def dof() -> int:
        return 6
    
    @staticmethod
    def mat_size() -> int:
        return 4
    
    @staticmethod
    def mat_adj_size() -> int:
        return 6
    
    def mat(self) -> Union[np.ndarray, jnp.ndarray]:
        batch_shape = self._rot.shape[:-2]
        if self.lib == 'jax':
            mat = jnp.zeros(batch_shape + (4, 4), dtype=self._rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(self._rot)
            mat = mat.at[..., 0:3, 3].set(self._pos)
            mat = mat.at[..., 3, 3].set(1)
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros(batch_shape + (4, 4), dtype=self._rot.dtype)
            mat[..., 0:3, 0:3] = self._rot
            mat[..., 0:3, 3] = self._pos
            mat[..., 3, 3] = 1
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
       
    
    @classmethod
    def set_mat(cls, mat = np.identity(4), LIB : str = 'numpy') -> 'SE3':
        mat = as_inexact(mat, LIB)
        return cls(mat[..., 0:3, 0:3], mat[..., 0:3, 3], LIB)
    
    @classmethod
    def set_pos_quaternion(cls, pos: Union[np.ndarray, jnp.ndarray],
                           quaternion: Union[np.ndarray, jnp.ndarray], LIB: str = 'numpy') -> 'SE3':
        pos, quaternion = as_inexact(pos, LIB), as_inexact(quaternion, LIB)
        if pos.ndim == 0 or pos.shape[-1] != 3:
            raise ValueError("Position must have shape (..., 3).")
        if quaternion.shape != (4,):
            raise ValueError("Quaternion must be a 4-element vector.")
        return cls(SO3.quaternion_to_mat(quaternion, LIB), pos, LIB)

    def pos(self ) -> Union[np.ndarray, jnp.ndarray]:
        return readonly_view(self._pos)

    def rot(self) -> Union[np.ndarray, jnp.ndarray]:
        return readonly_view(self._rot)

    def pos_quaternion(self) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]:
        return readonly_view(self._pos), SO3(self._rot, self.lib).quaternion()

    @classmethod
    def eye(cls, LIB : str = 'numpy') -> 'SE3':
        xp = array_lib(LIB)
        return cls(xp.identity(3), xp.zeros(3), LIB)

    def inv(self) -> 'SE3':
        rot_t = transpose_last(self._rot)
        return type(self)(rot_t, -matvec(rot_t, self._pos), self.lib)

    def mat_inv(self) -> Union[np.ndarray, jnp.ndarray]:
        rot_t = transpose_last(self._rot)
        pos = -matvec(rot_t, self._pos)
        batch_shape = self._rot.shape[:-2]
        if self.lib == 'jax':
            mat = jnp.zeros(batch_shape + (4, 4), dtype=self._rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(rot_t)
            mat = mat.at[..., 0:3, 3].set(pos)
            mat = mat.at[..., 3, 3].set(1)
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros(batch_shape + (4, 4), dtype=self._rot.dtype)
            mat[..., 0:3, 0:3] = rot_t
            mat[..., 0:3, 3] = pos
            mat[..., 3, 3] = 1
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        batch_shape = self._rot.shape[:-2]
        pos_hat_rot = SO3.hat(self._pos, self.lib) @ self._rot
        if self.lib == 'jax':
            mat = jnp.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(self._rot)
            mat = mat.at[..., 3:6, 0:3].set(pos_hat_rot)
            mat = mat.at[..., 3:6, 3:6].set(self._rot)
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat[..., 0:3, 0:3] = self._rot
            mat[..., 3:6, 0:3] = pos_hat_rot
            mat[..., 3:6, 3:6] = self._rot
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def set_mat_adj(mat = np.identity(6), LIB : str = 'numpy') -> 'SE3':
        
        mat = as_inexact(mat, LIB)
        rot = (mat[..., 0:3,0:3] + mat[..., 3:6,3:6]) * 0.5
        pos = SO3.vee(mat[..., 3:6,0:3] @ transpose_last(rot), LIB)
        
        return SE3(rot, pos, LIB)

    def mat_inv_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        rot_t = transpose_last(self._rot)
        lower = -rot_t @ SO3.hat(self._pos, self.lib)
        batch_shape = self._rot.shape[:-2]
        if self.lib == 'jax':
            mat = jnp.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(rot_t)
            mat = mat.at[..., 3:6, 0:3].set(lower)
            mat = mat.at[..., 3:6, 3:6].set(rot_t)
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat[..., 0:3, 0:3] = rot_t
            mat[..., 3:6, 0:3] = lower
            mat[..., 3:6, 3:6] = rot_t
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat operator on the tanget space vector
        mat =
        [ SO3.hat(w)  v ]
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        w, v = vec[..., 0:3], vec[..., 3:6]
        if LIB == "jax":
            mat = jnp.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(SO3.hat(w, LIB))
            mat = mat.at[..., 0:3, 3].set(v)
            return mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = SO3.hat(w, LIB)
            mat[..., 0:3, 3] = v
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
        
    @staticmethod
    def hat_commute(vec: Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat commute operator on the tanget space vector
        hat(a) @ b = hat_commute(b) @ a 
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] < 3:
            raise ValueError("Input vector must have at least 3 elements.")
        w = vec[..., 0:3]
        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (4, 6), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(SO3.hat(w, LIB))
            return -mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (4, 6), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = SO3.hat(w, LIB)
            return -mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def vee(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        a = vee(hat(a))
        '''
        vec_hat = as_inexact(vec_hat, LIB)
        if vec_hat.shape[-2:] != (4,4):
            raise ValueError("Input matrix must be of size (...,4,4).")
        
        if LIB == 'jax':
            w = SO3.vee(vec_hat[..., 0:3,0:3], LIB)
            v = vec_hat[..., 0:3,3]
            return jnp.concatenate((w, v), axis=-1)
        elif LIB == 'numpy':
            w = SO3.vee(vec_hat[..., 0:3,0:3], LIB)
            v = vec_hat[..., 0:3,3]
            return np.concatenate((w, v), axis=-1)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        rot, pos = vec[..., 0:3], vec[..., 3:6]
        R = SO3.exp(rot, a, LIB)
        V = SO3.exp_integ(rot, a, LIB)
        p = matvec(V, pos)
        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(R)
            mat = mat.at[..., 0:3, 3].set(p)
            mat = mat.at[..., 3, 3].set(1)
            return mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = R
            mat[..., 0:3, 3] = p
            mat[..., 3, 3] = 1
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    def log(self) -> Union[np.ndarray, jnp.ndarray]:
        """Return principal real exponential coordinates ``[..., (phi, rho)]``.

        The rotation has norm at most pi. At pi either axis sign can represent
        the rotation, and the logarithm is discontinuous across this branch.
        """
        if self._rot.dtype.kind == 'c':
            raise TypeError("SE3.log requires a real pose")
        xp = array_lib(self.lib)
        phi = SO3._log_mat(self._rot, self.lib)
        theta_squared = xp.sum(phi * phi, axis=-1)
        small = theta_squared < 1 / 64
        safe_squared = xp.where(small, xp.ones_like(theta_squared), theta_squared)
        half = xp.sqrt(safe_squared) / 2
        coefficient = (1 - half * xp.cos(half) / xp.sin(half)) / safe_squared
        series = 1 / 12 + theta_squared * (
            1 / 720 + theta_squared * (1 / 30240 + theta_squared / 1209600)
        )
        coefficient = xp.where(small, series, coefficient)

        # Apply the inverse of SO3.exp_integ(phi, 1) to the position.
        hat = SO3.hat(phi, self.lib)
        cross = matvec(hat, self._pos)
        rho = self._pos - cross / 2 + coefficient[..., None] * matvec(hat, cross)
        return xp.concatenate((phi, rho), axis=-1).astype(self._rot.dtype)

    def integrate(self, delta) -> 'SE3':
        """Return ``self @ Exp(delta)`` for a body-frame increment ``(..., 6)``."""
        delta = as_inexact(delta, self.lib)
        if delta.ndim == 0 or delta.shape[-1] != SE3.dof():
            raise ValueError("Increment must have shape (..., 6)")
        increment = SE3.set_mat(SE3.exp(delta, LIB=self.lib), self.lib)
        return self @ increment

    def difference(self, other: 'SE3') -> Union[np.ndarray, jnp.ndarray]:
        """Return ``Log(self.inv() @ other)`` in the body frame of ``self``."""
        if not isinstance(other, SE3):
            raise TypeError("SE3.difference requires an SE3 pose")
        if self.lib != other.lib:
            raise ValueError("SE3.difference requires matching backends")
        rotation_inverse = transpose_last(self._rot)
        relative_rot = rotation_inverse @ other._rot
        relative_pos = matvec(rotation_inverse, other._pos - self._pos)
        return SE3(relative_rot, relative_pos, self.lib).log()

    @staticmethod
    def __integ_p_cross_r(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """
            Compute the integral of hat(p) @ R.
        """
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        if LIB == 'numpy':
            theta = np.linalg.norm(vec[0:3])
            if not math.isclose(theta, 1.0):
                a_ = a*theta
            else:
                a_ = a

            if math.isclose(theta, 0.0):
                return 0.5*a*a*SO3.hat(vec[3:6])
            else:
                u, v, w = vec[0:3] / theta
                x, y, z = vec[3:6]
                k = 1. / (theta*theta)
        else:
            raise ValueError("Unsupported library. Choose 'numpy'.")

        sa = np.sin(a_)
        ca = np.cos(a_)

        mat = np.zeros((3,3))
        
        coeff1 = k*(2. - 2.*ca - 0.5*a_*sa)
        coeff2 = k*(2.*a_ - 2.5*sa + 0.5*a_*ca)
        coeff3 = k*(1. - ca - 0.5*a_*sa)
        coeff4 = k*(a_ - 1.5*sa + 0.5*a_*ca)
        
        ux = u*x
        uy = u*y 
        uz = u*z
        vx = v*x
        vy = v*y
        vz = v*z
        wx = w*x
        wy = w*y
        wz = w*z
        
        uu = u*u
        vv = v*v
        ww = w*w
        
        uy_vx = uy + vx
        uz_wx = uz + wx
        vz_wy = vz + wy
        
        ux_vy = ux + vy
        vy_wz = vy + wz
        wz_ux = wz + ux
        
        uu_vv = uu + vv
        vv_ww = vv + ww
        ww_uu = ww + uu
        
        uu_vv_ww = uu + vv + ww
        
        m00_2 = -2*vy_wz
        m10_2 = uy_vx
        m20_2 = uz_wx
        m11_2 = -2*wz_ux
        m21_2 = vz_wy
        m22_2 = -2*ux_vy
        
        m00_3 = u*v*z - u*w*y - v*m20_2 + w*m10_2
        m10_3 = -v*w*y - v*m21_2 + w*m11_2 + z*-ww_uu
        m20_3 = v*wz - v*m22_2 + w*m21_2 - y*-uu_vv
        m01_3 = w*ux+ u*m20_2 - w*m00_2 - z*-vv_ww
        m11_3 = -u*v*z + u*m21_2 + v*w*x - w*m10_2
        m21_3 = -u*wz + u*m22_2 - w*m20_2 + x*-uu_vv
        m02_3 = -v*ux - u*m10_2 + v*m00_2 + y*-vv_ww
        m12_3 = u*vy - u*m11_2 + v*m10_2 - x*-ww_uu
        m22_3 = u*w*y - u*m21_2 - v*w*x + v*m20_2
        
        mat[0,0] = coeff2 * m00_2 + coeff3 * m00_3 \
            + coeff4 * (-v*m02_3 + w*m01_3 + vy_wz*uu_vv_ww)

        mat[1,0] = coeff1 * z + coeff2 * m10_2 + coeff3 * m10_3 \
            + coeff4 * (-v*m12_3 + w*m11_3 - uy*uu_vv_ww)
        
        mat[2,0] = coeff1 * -y + coeff2 * m20_2 + coeff3 * m20_3 \
            + coeff4 * (-v*m22_3 + w*m21_3 - uz*uu_vv_ww)

        mat[0,1] = coeff1 * -z + coeff2 * m10_2 + coeff3 * m01_3 \
            + coeff4 * (u*m02_3 - w*m00_3 - vx*uu_vv_ww)

        mat[1,1] = coeff2 * m11_2 + coeff3 * m11_3 \
            + coeff4 * (u*m12_3 - w*m10_3 + wz_ux*uu_vv_ww)
        
        mat[2,1] = coeff1 * x + coeff2 * m21_2 + coeff3 * m21_3 \
            + coeff4 * (u*m22_3 - w*m20_3 - vz*uu_vv_ww)
        
        mat[0,2] = coeff1 * y + coeff2 * m20_2 + coeff3 * m02_3 \
            + coeff4 * (-u*m01_3 + v*m00_3 - wx*uu_vv_ww)

        mat[1,2] = coeff1 * -x + coeff2 * m21_2 + coeff3 * m12_3 \
            + coeff4 * (-u*m11_3 + v*m10_3 - wy*uu_vv_ww)
        
        mat[2,2] = coeff2 * m22_2 + coeff3 * m22_3 \
            + coeff4 * (-u*m21_3 + v*m20_3 + ux_vy*uu_vv_ww)

        return mat

    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Integrate exp(hat(vec) * t) from zero to a (bottom-right = a)."""
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        rot = vec[..., 0:3]
        pos = vec[..., 3:6]
        R = SO3.exp_integ(rot, a, LIB)
        V = SO3.exp_integ2nd(rot, a, LIB)
        p = matvec(V, pos)

        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(R)
            mat = mat.at[..., 0:3, 3].set(p)
            mat = mat.at[..., 3, 3].set(a)
            return mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (4, 4), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = R
            mat[..., 0:3, 3] = p
            mat[..., 3, 3] = a
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat operator on the tanget space vector for adjoint representation
        mat =
        [ SO3.hat(w)      0       ]
        [ SO3.hat(v)  SO3.hat(w)  ]
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        w, v = vec[..., :3], vec[..., 3:]
        w_hat = SO3.hat(w, LIB)
        v_hat = SO3.hat(v, LIB)

        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(w_hat)
            mat = mat.at[..., 3:6, 0:3].set(v_hat)
            mat = mat.at[..., 3:6, 3:6].set(w_hat)
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = w_hat
            mat[..., 3:6, 0:3] = v_hat
            mat[..., 3:6, 3:6] = w_hat
        else:
            raise ValueError("Unsupported library. Choose 'numpy', 'jax'.")

        return mat
    
    @staticmethod
    def hat_commute_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
            hat_adj(a) @ b  = hat_commute_adj(a) @ b
            return -hat_adj(vec)
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        return -SE3.hat_adj(vec, LIB)

    @staticmethod
    def vee_adj(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec_hat = as_inexact(vec_hat, LIB)
        if vec_hat.shape[-2:] != (6,6):
            raise ValueError("Input matrix must be of size (...,6,6).")
        
        if LIB == 'jax':
            w = 0.5 * ( SO3.vee(vec_hat[..., 0:3,0:3], LIB) + SO3.vee(vec_hat[..., 3:6,3:6], LIB) )
            v = SO3.vee(vec_hat[..., 3:6,0:3], LIB)
            return jnp.concatenate((w, v), axis=-1)
        elif LIB == 'numpy':
            w = 0.5 * (SO3.vee(vec_hat[..., 0:3, 0:3], LIB) + SO3.vee(vec_hat[..., 3:6, 3:6], LIB))
            v = SO3.vee(vec_hat[..., 3:6, 0:3], LIB)
            return np.concatenate([w, v], axis=-1)
        else:
            raise ValueError("Unsupported library. Choose 'numpy', 'jax'.")
    
    @staticmethod
    def exp_adj(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        Compute the adjoint representation of SE3.exp(vec, a).
        The angular component vec[..., :3] need not have unit norm.
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")

        h = SE3.exp(vec, a, LIB)

        rot = h[..., 0:3, 0:3]
        pos_hat_rot = SO3.hat(h[..., 0:3, 3], LIB) @ rot
        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (6, 6), dtype=rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(rot)
            mat = mat.at[..., 3:6, 0:3].set(pos_hat_rot)
            mat = mat.at[..., 3:6, 3:6].set(rot)
            return mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (6, 6), dtype=rot.dtype)
            mat[..., 0:3, 0:3] = rot
            mat[..., 3:6, 0:3] = pos_hat_rot
            mat[..., 3:6, 3:6] = rot
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def exp_integ_adj(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        return a * SE3._unit_adjoint_integral(a * vec, (), LIB)

    @staticmethod
    def right_jacobian(xi, LIB: str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Return the exponential Jacobian in body coordinates, with shape (..., 6, 6)."""
        return SE3._unit_adjoint_integral(-as_inexact(xi, LIB), (), LIB)

    @staticmethod
    def right_jacobian_directional(xi, u, LIB: str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Return DJ_r(xi)[u] for a direction in the same exponential coordinates."""
        return SE3._unit_adjoint_integral(-as_inexact(xi, LIB), (-as_inexact(u, LIB),), LIB)

    @staticmethod
    def right_jacobian_second_directional(xi, u, v, LIB: str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Return the symmetric mixed directional derivative D²J_r(xi)[u, v]."""
        directions = (-as_inexact(u, LIB), -as_inexact(v, LIB))
        return SE3._unit_adjoint_integral(-as_inexact(xi, LIB), directions, LIB)

    @staticmethod
    def _right_jacobian_series(coordinates, LIB):
        if coordinates.shape[-2] == 1:
            return SE3.right_jacobian(coordinates[..., 0, :], LIB)[..., None, :, :]
        return adjoint_integral_series(coordinates[..., :3], SE3.hat_adj(-coordinates, LIB), array_lib(LIB))

    @staticmethod
    def _unit_adjoint_integral(vec, directions, LIB):
        """Evaluate the adjoint integral over [0, 1] or its directional derivatives."""
        xp = array_lib(LIB)
        inputs = (vec, *directions)
        if any(value.ndim == 0 or value.shape[-1] != 6 for value in inputs):
            raise ValueError("Input vector and directions must have shape (..., 6)")
        order = len(directions)
        dtype = xp.result_type(*(value.dtype for value in inputs))
        vec, *directions = (value.astype(dtype, copy=False) for value in inputs)
        angular = vec[..., :3]
        theta_squared = xp.sum(angular * angular, axis=-1)
        coefficients = adjoint_integral_coefficients(theta_squared, order, xp)
        generator = SE3.hat_adj(vec, LIB)
        identity = xp.eye(6, dtype=dtype)

        if order == 0:
            a1, a2, a3, a4 = coefficients[0]
            polynomial = a4[..., None, None] * generator + a3[..., None, None] * identity
            polynomial = polynomial @ generator + a2[..., None, None] * identity
            polynomial = polynomial @ generator + a1[..., None, None] * identity
            return identity + polynomial @ generator

        u = directions[0]
        generator_u = SE3.hat_adj(u, LIB)
        theta_u = 2 * xp.sum(angular * u[..., :3], axis=-1)
        if order == 2:
            v = directions[1]
            generator_v = SE3.hat_adj(v, LIB)
            theta_v = 2 * xp.sum(angular * v[..., :3], axis=-1)
            theta_uv = 2 * xp.sum(u[..., :3] * v[..., :3], axis=-1)

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
                value_uv = (value_uv @ generator + value_u @ generator_v
                            + value_v @ generator_u + dc_uv[..., None, None] * identity)
                value_v = (value_v @ generator + value @ generator_v
                           + (dc * theta_v)[..., None, None] * identity)
            value_u = (value_u @ generator + value @ generator_u
                       + (dc * theta_u)[..., None, None] * identity)
            value = value @ generator + c[..., None, None] * identity
        if order == 2:
            return value_uv @ generator + value_u @ generator_v + value_v @ generator_u
        return value_u @ generator + value @ generator_u

    @staticmethod
    def sub_tan_vec(val0 : 'SE3', val1 : 'SE3', 
                    frame : str = 'bframe', LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:

        w = SO3.sub_tan_vec(SO3(val0.rot(),LIB), SO3(val1.rot(),LIB), frame, LIB)

        check_frame(frame)
        if frame == 'bframe':
            v = matvec(transpose_last(val0.rot()), (val1.pos() - val0.pos()))
        elif frame == 'fframe':
            tmp = (val1.rot() - val0.rot()) @ transpose_last(val0.rot())
            v = (val1.pos() - val0.pos()) - matvec(tmp, val0.pos())
        
        if LIB == 'numpy':
            vec = np.concatenate([w, v], axis=-1)
        elif LIB == 'jax':
            vec = jnp.concatenate([w, v], axis=-1)

        return vec

    def se3_mul(l_rot : Union[np.ndarray, jnp.ndarray], 
                l_pos : Union[np.ndarray, jnp.ndarray], 
                r_rot : Union[np.ndarray, jnp.ndarray], 
                r_pos : Union[np.ndarray, jnp.ndarray]) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]:
        assert isinstance(l_rot, jnp.ndarray) or isinstance(l_rot, np.ndarray), "Input must be a numpy or jax array."
        assert isinstance(l_pos, jnp.ndarray) or isinstance(l_pos, np.ndarray), "Input must be a numpy or jax array."
        assert isinstance(r_rot, jnp.ndarray) or isinstance(r_rot, np.ndarray), "Input must be a numpy or jax array."
        assert isinstance(r_pos, jnp.ndarray) or isinstance(r_pos, np.ndarray), "Input must be a numpy or jax array."
        return SO3.so3_mul(l_rot, r_rot), l_pos + matvec(l_rot, r_pos)
    
    def __matmul__(self, rval):
        if isinstance(rval, SE3):
            rot, pos = SE3.se3_mul(self._rot, self._pos, rval._rot, rval._pos)
            return type(self)(rot, pos, self.lib)
        elif isinstance(rval, (np.ndarray, jnp.ndarray)):
            if rval.shape[-1] == 3 and (rval.ndim == 1 or rval.shape[-2:] != (4, 4)):
                return matvec(self._rot, rval) + self._pos
            elif rval.shape[-1] == 6 and (rval.ndim == 1 or rval.shape[-2:] != (6, 6)):
                rot_part = matvec(self._rot, rval[..., 0:3])
                pos_part = matvec(SO3.hat(self._pos, self.lib) @ self._rot, rval[..., 0:3]) + matvec(self._rot, rval[..., 3:6])
                if self.lib == 'jax' or isinstance(rval, jnp.ndarray):
                    return jnp.concatenate((rot_part, pos_part), axis=-1)
                return np.concatenate((rot_part, pos_part), axis=-1)
            elif rval.shape[-2:] == (4,4):
                return self.mat() @ rval
            elif rval.shape[-2:] == (6,6):
                return self.mat_adj() @ rval
            else:
                raise TypeError("Right operand has unsupported shape")
        else:
            raise TypeError("Right operand should be SE3, numpy.ndarray, or jax.ndarray")

    @classmethod
    def rand(cls, LIB = 'numpy') -> 'SE3':
        if LIB == 'jax':
            p = jax.random.uniform(jax.random.PRNGKey(0), (3,))
        elif LIB == 'numpy':
            p = np.random.rand(3) 
        return cls(cls._cls_so3.rand(LIB).mat(), p, LIB)
    
    def __repr__(self):
        return f"SE3(\nrot=\n{self._rot},\npos=\n{self._pos},\nLIB='{self.lib}')"
    
    @classmethod
    def change_class(cls, a):
        b = cls.__new__(cls)
        b.__dict__ = a.__dict__.copy()
        cls.__init__(b, a.rot(), a.pos(), a.lib)
        return b

class SE3wrench(SE3):
    """SE3 pose acting on wrenches: ``mat_adj`` and ``hat_adj`` are coadjoint.

    ``mat``, ``hat`` and ``vee`` are those of SE3, so motion composition (for
    example in CMTM products) uses ``_motion_cls`` rather than ``hat_adj``.
    """
    _motion_cls = SE3

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        upper = SO3.hat(self._pos, self.lib) @ self._rot
        batch_shape = self._rot.shape[:-2]
        if self.lib == 'jax':
            mat = jnp.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(self._rot)
            mat = mat.at[..., 0:3, 3:6].set(upper)
            mat = mat.at[..., 3:6, 3:6].set(self._rot)
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat[..., 0:3, 0:3] = self._rot
            mat[..., 0:3, 3:6] = upper
            mat[..., 3:6, 3:6] = self._rot
            return mat
    
    def mat_inv_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        rot_t = transpose_last(self._rot)
        upper = -rot_t @ SO3.hat(self._pos, self.lib)
        batch_shape = self._rot.shape[:-2]
        if self.lib == 'jax':
            mat = jnp.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat = mat.at[..., 0:3, 0:3].set(rot_t)
            mat = mat.at[..., 0:3, 3:6].set(upper)
            mat = mat.at[..., 3:6, 3:6].set(rot_t)
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros(batch_shape + (6, 6), dtype=self._rot.dtype)
            mat[..., 0:3, 0:3] = rot_t
            mat[..., 0:3, 3:6] = upper
            mat[..., 3:6, 3:6] = rot_t
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SE3.exp_adj(vec, -a, LIB))
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -transpose_last(SE3.exp_integ_adj(vec, -a, LIB))
    
    @staticmethod
    def exp_adj(vec, a=1., LIB='numpy'):
        """Wrench action consistent with mat_adj and hat_adj."""
        return SE3wrench.exp(vec, a, LIB)

    @staticmethod
    def exp_integ_adj(vec, a, LIB='numpy'):
        """Integral of the wrench action."""
        return SE3wrench.exp_integ(vec, a, LIB)

    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        mat = 
        [ SO3.hat(w)   SO3.hat(v) ]
        [     0        SO3.hat(w) ]
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        w, v = vec[..., :3], vec[..., 3:]
        w_hat = SO3.hat(w, LIB)
        v_hat = SO3.hat(v, LIB)

        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(w_hat)
            mat = mat.at[..., 0:3, 3:6].set(v_hat)
            mat = mat.at[..., 3:6, 3:6].set(w_hat)
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = w_hat
            mat[..., 0:3, 3:6] = v_hat
            mat[..., 3:6, 3:6] = w_hat
        else:
            raise ValueError("Unsupported library. Choose 'numpy', 'jax'.")

        return mat
    
    @staticmethod
    def hat_commute(vec: Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat commute operator on the tanget space vector
        hat(a) @ b = hat_commute(b) @ a 
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] < 3:
            raise ValueError("Input vector must have at least 3 elements.")
        w = vec[..., 0:3]
        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (4, 6), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(SO3.hat(w, LIB))
            return -mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (4, 6), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = SO3.hat(w, LIB)
            return -mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def vee_adj(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """Inverse of hat_adj: the force part sits in the upper-right block."""
        vec_hat = as_inexact(vec_hat, LIB)
        if vec_hat.shape[-2:] != (6, 6):
            raise ValueError("Input matrix must be of size (...,6,6).")
        w = 0.5 * (SO3.vee(vec_hat[..., 0:3, 0:3], LIB) + SO3.vee(vec_hat[..., 3:6, 3:6], LIB))
        v = SO3.vee(vec_hat[..., 0:3, 3:6], LIB)
        return array_lib(LIB).concatenate((w, v), axis=-1)

    @staticmethod
    def hat_commute_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
            hat_adj(a) @ b  = hat_commute_adj(a) @ b
            mat =
            [[ -SO3.hat(w)     -SO3.hat(v) ]
             [ -SO3.hat(v)     -SO3.hat(w) ]]
        '''
        vec = as_inexact(vec, LIB)
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")

        if LIB == 'jax':
            mat = jnp.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
            mat = mat.at[..., 0:3, 0:3].set(SO3.hat(vec[..., 0:3], LIB))
            mat = mat.at[..., 0:3, 3:6].set(SO3.hat(vec[..., 3:6], LIB))
            mat = mat.at[..., 3:6, 0:3].set(SO3.hat(vec[..., 3:6], LIB))
            return -mat
        elif LIB == 'numpy':
            mat = np.zeros(vec.shape[:-1] + (6, 6), dtype=vec.dtype)
            mat[..., 0:3, 0:3] = SO3.hat(vec[..., 0:3], LIB)
            mat[..., 0:3, 3:6] = SO3.hat(vec[..., 3:6], LIB)
            mat[..., 3:6, 0:3] = SO3.hat(vec[..., 3:6], LIB)
            return -mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    def __repr__(self):
        return f"SE3wrench(\nrot=\n{self._rot},\npos=\n{self._pos},\nLIB='{self.lib}')"
    
'''
    Khalil, et al. 1995
'''
class SE3inertia(SE3):
    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        mpg = vec[1:4]
        if LIB == 'jax':
            return jnp.block([
                [SO3inertia.hat(vec[4:10], LIB), SO3wrench.hat(mpg, LIB)],
                [SO3.hat(mpg, LIB), vec[0] * jnp.identity(3, dtype=vec.dtype)]
            ])
        elif LIB == 'numpy':
            mat = np.zeros((6,6), dtype=vec.dtype)
            mat[0:3,0:3] = SO3inertia.hat(vec[4:10], LIB)
            mat[0:3,3:6] = SO3wrench.hat(mpg, LIB)
            mat[3:6,0:3] = SO3.hat(mpg, LIB)
            mat[3:6,3:6] = vec[0]*np.identity(3)
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        vec = as_inexact(vec, LIB)
        v = vec[3:6]
        w = vec[0:3]
        if LIB == 'jax':
            top = jnp.concatenate((
                jnp.zeros((3, 1), dtype=vec.dtype),
                SO3wrench.hat_commute(v, LIB),
                SO3inertia.hat_commute(w, LIB)
            ), axis=1)
            bottom = jnp.concatenate((
                v.reshape(3, 1),
                SO3.hat_commute(w, LIB),
                jnp.zeros((3, 6), dtype=vec.dtype)
            ), axis=1)
            return jnp.concatenate((top, bottom), axis=0)
        elif LIB == 'numpy':
            mat = np.zeros((6,10), dtype=vec.dtype)
            mat[3:6,0] = v
            mat[0:3,1:4] = SO3wrench.hat_commute(v, LIB)
            mat[3:6,1:4] = SO3.hat_commute(w, LIB)
            mat[0:3,4:10] = SO3inertia.hat_commute(w, LIB)
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    def __repr__(self):
        return f"SE3inertia(\nrot=\n{self._rot},\npos=\n{self._pos},\nLIB='{self.lib}')"
