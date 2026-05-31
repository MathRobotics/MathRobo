from typing import Union, Tuple
import math

import jax
import numpy as np
import jax.numpy as jnp

from .._batch import array_lib, matvec, transpose_last
from .lie_abst import LieAbstract
from .so3 import SO3, SO3inertia, SO3wrench

class SE3(LieAbstract):
    _dof = 6
    _cls_so3 = SO3
    def __init__(self, rot = np.identity(3), pos = np.zeros(3), LIB : str = 'numpy'): 
        '''
        Constructor
        '''
        self._rot = rot
        self._pos = pos
        self._lib = LIB

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
       
    
    @staticmethod
    def set_mat(mat = np.identity(4), LIB : str = 'numpy') -> 'SE3':
        return SE3(mat[..., 0:3, 0:3], mat[..., 0:3, 3], LIB)
    
    @staticmethod
    def set_pos_quaternion(pos: Union[np.ndarray, jnp.ndarray], 
                           quaternion: Union[np.ndarray, jnp.ndarray], LIB: str = 'numpy') -> 'SE3':
        assert len(pos) == 3, "Position must be a 3-element vector."
        assert len(quaternion) == 4, "Quaternion must be a 4-element vector."
        assert isinstance(pos, (np.ndarray, jnp.ndarray)), "Position must be a numpy or jax array."
        assert isinstance(quaternion, (np.ndarray, jnp.ndarray)), "Quaternion must be a numpy or jax array."
        return SE3(SO3.quaternion_to_mat(quaternion, LIB), pos, LIB)

    def pos(self ) -> Union[np.ndarray, jnp.ndarray]:
        return self._pos

    def rot(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._rot

    def pos_quaternion(self) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]:
        return self._pos, SO3.quaternion(SO3.set_mat(self._rot, self.lib))

    @staticmethod
    def eye(LIB : str = 'numpy'):
        if LIB == 'jax':
            return SE3(jnp.identity(3), jnp.zeros(3), LIB)
        elif LIB == 'numpy':
          return SE3(np.identity(3), np.zeros(3), LIB)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    def inv(self) -> 'SE3':
        rot_t = transpose_last(self._rot)
        return SE3(rot_t, -matvec(rot_t, self._pos), self.lib)

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
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        xp = array_lib(LIB)
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
    
    @staticmethod
    def __integ_p_cross_r(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """
            p x Rの積分の計算
        """
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
        '''
        vec[0:3]の大きさは1を想定
        '''
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

    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat operator on the tanget space vector for adjoint representation
        mat =
        [ SO3.hat(w)      0       ]
        [ SO3.hat(v)  SO3.hat(w)  ]
        '''
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
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        return -SE3.hat_adj(vec, LIB)

    @staticmethod
    def vee_adj(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
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
        SE3の随伴表現の計算
        vec[0:3]の大きさは1を想定
        '''
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
        if vec.shape[-1] != 6:
            raise ValueError("Input vector must be of size 6.")
        
        """
            SE3の随伴表現の積分の計算
        """
        xp = array_lib(LIB)
        w = vec[..., 0:3]
        n = xp.linalg.norm(w, axis=-1)
        a_ = a * n
        ca = xp.cos(a_)
        sa = xp.sin(a_)
        K = SE3.hat_adj(vec, LIB)
        K2 = K @ K
        K3 = K2 @ K
        K4 = K2 @ K2
        n_safe = xp.where(n == 0.0, 1.0, n)
        n2 = n_safe * n_safe
        n3 = n2 * n_safe
        n4 = n2 * n2
        n5 = n4 * n_safe
        A1 = xp.where(n == 0.0, 0.5 * a * a, 0.5 * (4.0 - 4.0 * ca - a_ * sa) / n2)
        A2 = xp.where(n == 0.0, 0.0, 0.5 * (4.0 * a_ - 5.0 * sa + a_ * ca) / n3)
        A3 = xp.where(n == 0.0, 0.0, 0.5 * (2.0 - 2.0 * ca - a_ * sa) / n4)
        A4 = xp.where(n == 0.0, 0.0, 0.5 * (2.0 * a_ - 3.0 * sa + a_ * ca) / n5)
        I = xp.eye(6, dtype=vec.dtype)
        return (
            a * I
            + A1[..., None, None] * K
            + A2[..., None, None] * K2
            + A3[..., None, None] * K3
            + A4[..., None, None] * K4
        )

    @staticmethod
    def sub_tan_vec(val0 : 'SE3', val1 : 'SE3', 
                    frame : str = 'bframe', LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:

        w = SO3.sub_tan_vec(SO3(val0.rot(),LIB), SO3(val1.rot(),LIB), frame, LIB)

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
            return SE3(rot, pos, self.lib)
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
    @staticmethod
    def set_mat(mat = np.identity(4), LIB : str = 'numpy') -> 'SE3wrench':
        return SE3wrench(mat[..., 0:3, 0:3], mat[..., 0:3, 3], LIB)
    
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
    
    def inv(self) -> 'SE3wrench':
        rot_t = transpose_last(self._rot)
        return SE3wrench(rot_t, -matvec(rot_t, self._pos), self.lib)
        
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
        return transpose_last(SE3.exp_adj(vec, a, LIB))
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SE3.exp_integ_adj(vec, a, LIB))
    
    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        mat = 
        [ SO3.hat(w)   SO3.hat(v) ]
        [     0        SO3.hat(w) ]
        '''
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
    def hat_commute_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
            hat_adj(a) @ b  = hat_commute_adj(a) @ b
            mat =
            [[ -SO3.hat(w)     -SO3.hat(v) ]
             [ -SO3.hat(v)     -SO3.hat(w) ]]
        '''
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
