from .lie_abst import *

from typing import Union

import jax
from .._batch import array_lib, matvec, transpose_last

class SO3(LieAbstract):
    _dof = 3
    def __init__(self, r = np.identity(3), LIB : str = 'numpy'):
        '''
        Constructor
        '''
        self._rot = r
        self._lib = LIB

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
        return self._rot
    
    @staticmethod
    def set_mat(mat = np.identity(3), LIB : str = 'numpy') -> 'SO3':
        return SO3(mat, LIB)
    
    def quaternion(self) -> np.ndarray:
        # trace
        trace = self._rot[0, 0] + self._rot[1, 1] + self._rot[2, 2]

        # (w, x, y, z)
        q = np.zeros(4, dtype=float)

        if trace > 0.0:
                # trace > 0
                s = 0.5 / np.sqrt(trace + 1.0)
                q[0] = 0.25 / s  # w
                q[1] = (self._rot[2, 1] - self._rot[1, 2]) * s  # x
                q[2] = (self._rot[0, 2] - self._rot[2, 0]) * s  # y
                q[3] = (self._rot[1, 0] - self._rot[0, 1]) * s  # z
        else:
                # trace <= 0
                # search maximum element in matrix diagonal
                if (self._rot[0, 0] > self._rot[1, 1]) and (self._rot[0, 0] > self._rot[2, 2]):
                        # self._rot[0, 0] is maximize
                        s = 2.0 * np.sqrt(1.0 + self._rot[0, 0] - self._rot[1, 1] - self._rot[2, 2])
                        q[0] = (self._rot[2, 1] - self._rot[1, 2]) / s  # w
                        q[1] = 0.25 * s                 # x
                        q[2] = (self._rot[0, 1] + self._rot[1, 0]) / s  # y
                        q[3] = (self._rot[0, 2] + self._rot[2, 0]) / s  # z
                elif self._rot[1, 1] > self._rot[2, 2]:
                        # self._rot[1, 1] is maximize
                        s = 2.0 * np.sqrt(1.0 + self._rot[1, 1] - self._rot[0, 0] - self._rot[2, 2])
                        q[0] = (self._rot[0, 2] - self._rot[2, 0]) / s  # w
                        q[1] = (self._rot[0, 1] + self._rot[1, 0]) / s  # x
                        q[2] = 0.25 * s                 # y
                        q[3] = (self._rot[1, 2] + self._rot[2, 1]) / s  # z
                else:
                        # self._rot[2, 2] is maximize
                        s = 2.0 * np.sqrt(1.0 + self._rot[2, 2] - self._rot[0, 0] - self._rot[1, 1])
                        q[0] = (self._rot[1, 0] - self._rot[0, 1]) / s  # w
                        q[1] = (self._rot[0, 2] + self._rot[2, 0]) / s  # x
                        q[2] = (self._rot[1, 2] + self._rot[2, 1]) / s  # y
                        q[3] = 0.25 * s                 # z

        # normalize
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-15:
                q /= q_norm

        return q  # [w, x, y, z]
        
    @staticmethod
    def quaternion_to_mat(quaternion : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
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

    @staticmethod
    def set_quaternion(quaternion : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> 'SO3':
        assert len(quaternion) == 4, "Quaternion must be a 4-element vector."
        assert isinstance(quaternion, (np.ndarray, jnp.ndarray)), "Quaternion must be a numpy or jax array."
        return SO3(SO3.quaternion_to_mat(quaternion, LIB), LIB)
    
    @staticmethod
    def set_euler(euler : Union[np.ndarray, jnp.ndarray], order : str = 'ZYX', LIB : str = 'numpy') -> 'SO3':
        assert len(euler) == 3, "Euler angles must be a 3-element vector."
        assert isinstance(euler, (np.ndarray, jnp.ndarray)), "Euler angles must be a numpy or jax array."
        roll, pitch, yaw = euler
        if LIB == 'jax':
            cr = jnp.cos(roll)
            sr = jnp.sin(roll)
            cp = jnp.cos(pitch)
            sp = jnp.sin(pitch)
            cy = jnp.cos(yaw)
            sy = jnp.sin(yaw)
            if order == 'ZYX':
                m = jnp.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [  -sp,           cp*sr,           cp*cr]
                ])
            elif order == 'ZXY':
                m = jnp.array([
                    [cy*cp + sy*sp*sr, -cy*sp + sy*cp*sr, sy*cr],
                    [sy*cp - cy*sp*sr, -sy*sp - cy*cp*sr, -cy*cr],
                    [        -cp*sr,             cp*cr,     sp]
                ])
            elif order == 'YXZ':
                m = jnp.array([
                    [cp*cy + sp*sr*sy, -cr*sy, sp*cy - cp*sr*sy],
                    [cp*sy - sp*sr*cy, cr*cy, sp*sy + cp*sr*cy],
                    [        -sp*cr,     sr,             cp*cr]
                ])
            elif order == 'YZX':
                m = jnp.array([
                    [cp*cy, sr*sp - cr*cy*sy, cr*sp + sr*cy*sy],
                    [   sp,           sr*cp,           cr*cp],
                    [-sy*cp, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy]
                ])
            elif order == 'XYZ':
                m = jnp.array([
                    [cp*cy, -cp*sy, sp],
                    [sr*sp*cy + cr*sy, -sr*sp*sy + cr*cy, -sr*cp],
                    [-cr*sp*cy + sr*sy, cr*sp*sy + sr*cy, cr*cp]
                ])
            elif order == 'XZY':
                m = jnp.array([
                    [cp*cy, -sy, sp*cy],
                    [sr*sp + cr*cp*sy, cr*cy, -sr*cp + cr*sp*sy],
                    [-cr*sp + sr*cp*sy, sr*cy, cr*cp + sr*sp*sy]
                ])
            else:
                raise ValueError("Unsupported order. Choose from 'ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY'.")
            return SO3(m, LIB)
        elif LIB == 'numpy':
            cr = np.cos(roll)
            sr = np.sin(roll)
            cp = np.cos(pitch)
            sp = np.sin(pitch)
            cy = np.cos(yaw)
            sy = np.sin(yaw)
            if order == 'ZYX':
                m = np.array([
                    [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                    [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                    [  -sp,           cp*sr,           cp*cr]
                ])
            elif order == 'ZXY':
                m = np.array([
                    [cy*cp + sy*sp*sr, -cy*sp + sy*cp*sr, sy*cr],
                    [sy*cp - cy*sp*sr, -sy*sp - cy*cp*sr, -cy*cr],
                    [        -cp*sr,             cp*cr,     sp]
                ])
            elif order == 'YXZ':
                m = np.array([
                    [cp*cy + sp*sr*sy, -cr*sy, sp*cy - cp*sr*sy],
                    [cp*sy - sp*sr*cy, cr*cy, sp*sy + cp*sr*cy],
                    [        -sp*cr,     sr,             cp*cr]
                ])
            elif order == 'YZX':
                m = np.array([
                    [cp*cy, sr*sp - cr*cy*sy, cr*sp + sr*cy*sy],
                    [   sp,           sr*cp,           cr*cp],
                    [-sy*cp, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy]
                ])
            elif order == 'XYZ':
                m = np.array([
                    [cp*cy, -cp*sy, sp],
                    [sr*sp*cy + cr*sy, -sr*sp*sy + cr*cy, -sr*cp],
                    [-cr*sp*cy + sr*sy, cr*sp*sy + sr*cy, cr*cp]
                ])
            elif order == 'XZY':
                m = np.array([
                    [cp*cy, -sy, sp*cy],
                    [sr*sp + cr*cp*sy, cr*cy, -sr*cp + cr*sp*sy],
                    [-cr*sp + sr*cp*sy, sr*cy, cr*cp + sr*sp*sy]
                ])
            else:
                raise ValueError("Unsupported order. Choose from 'ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY'.")
            return SO3(m, LIB)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def mat_to_quaternion(mat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        m = SO3(mat, LIB)
        return m.quaternion()
    
    @staticmethod
    def eye(LIB : str = 'numpy') -> 'SO3':
        if LIB == 'jax':
            return SO3(jnp.identity(3), LIB)
        elif LIB == 'numpy':
            return SO3(np.identity(3), LIB)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    def inv(self) -> 'SO3':
        return SO3(transpose_last(self._rot), self.lib)

    def mat_inv(self) -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(self._rot)

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._rot
    
    @staticmethod
    def set_mat_adj(mat = np.identity(3), LIB : str = 'numpy') -> 'SO3':
        return SO3(mat, LIB)

    def mat_inv_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(self._rot)

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
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
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        theta = xp.linalg.norm(vec, axis=-1)
        a_theta = a * theta
        K = SO3.hat(vec, LIB)
        K2 = K @ K
        theta_safe = xp.where(theta == 0, 1.0, theta)
        theta2 = theta_safe * theta_safe
        A = xp.where(theta == 0, a, xp.sin(a_theta) / theta_safe)
        B = xp.where(theta == 0, 0.0, (1.0 - xp.cos(a_theta)) / theta2)
        I = xp.eye(3, dtype=vec.dtype)
        return I + A[..., None, None] * K + B[..., None, None] * K2
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        theta = xp.linalg.norm(vec, axis=-1)
        a_theta = a * theta
        K = SO3.hat(vec, LIB)
        K2 = K @ K
        theta_safe = xp.where(theta == 0, 1.0, theta)
        theta2 = theta_safe * theta_safe
        theta3 = theta2 * theta_safe
        A = xp.where(theta == 0, 0.0, (1.0 - xp.cos(a_theta)) / theta2)
        B = xp.where(theta == 0, 0.0, (a_theta - xp.sin(a_theta)) / theta3)
        I = xp.eye(3, dtype=vec.dtype)
        return a * I + A[..., None, None] * K + B[..., None, None] * K2
    
    @staticmethod
    def exp_integ2nd(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if vec.shape[-1] != 3:
            raise ValueError("Input vector must be of size 3.")
        xp = array_lib(LIB)
        theta = xp.linalg.norm(vec, axis=-1)
        a_theta = a * theta
        K = SO3.hat(vec, LIB)
        K2 = K @ K
        theta_safe = xp.where(theta == 0, 1.0, theta)
        theta2 = theta_safe * theta_safe
        theta3 = theta2 * theta_safe
        theta4 = theta2 * theta2
        A = xp.where(theta == 0, 0.0, (a_theta - xp.sin(a_theta)) / theta3)
        B = xp.where(theta == 0, 0.0, (0.5 * a_theta * a_theta - 1.0 + xp.cos(a_theta)) / theta4)
        I = xp.eye(3, dtype=vec.dtype)
        return 0.5 * a * a * I + A[..., None, None] * K + B[..., None, None] * K2
    
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

    @staticmethod
    def sub_tan_vec(val0 : 'SO3', val1 : 'SO3', 
                    frame : str = 'bframe', LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if frame == 'bframe':
            vec = SO3.vee(val0.mat_inv() @ (val1._rot - val0._rot), LIB)
        elif frame == 'fframe':
            vec = SO3.vee((val1._rot - val0._rot) @ val0.mat_inv(), LIB)
        return vec
    
    @staticmethod
    def so3_mul(l_rot : Union[np.ndarray, jnp.ndarray], r_rot : Union[np.ndarray, jnp.ndarray]) -> Union[np.ndarray, jnp.ndarray]:
        assert isinstance(l_rot, jnp.ndarray) or isinstance(l_rot, np.ndarray), "Input must be a numpy or jax array."
        assert isinstance(r_rot, jnp.ndarray) or isinstance(r_rot, np.ndarray), "Input must be a numpy or jax array."
        return l_rot @ r_rot

    def __matmul__(self, rval):
        if isinstance(rval, SO3):
            return SO3(SO3.so3_mul(self._rot, rval._rot), self.lib)
        elif isinstance(rval, (np.ndarray, jnp.ndarray)):
            if rval.shape[-1] == 3 and (rval.ndim == 1 or rval.shape[-2:] != (3, 3)):
                return matvec(self._rot, rval)
            return SO3.so3_mul(self._rot, rval)
        else:
            raise TypeError("Right operand should be SO3, numpy.ndarray, or jax.ndarray")

    @classmethod
    def rand(cls, LIB : str = 'numpy') -> 'SO3':
        if LIB == 'jax':
            v = jax.random.uniform(jax.random.PRNGKey(0), (3,))
            m = SO3.exp(v, LIB='jax')
            return cls(m, LIB)
        elif LIB == 'numpy':
            v = np.random.rand(3) 
            m = SO3.exp(v)
            return cls(m, LIB)

    def __repr__(self):
        return f"SO3(\nrot=\n{self._rot},\nLIB='{self.lib}')"
    
class SO3wrench(SO3):
    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.hat(vec, LIB)
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.hat(vec, LIB)

    def inv(self) -> 'SO3wrench':
        return SO3wrench(transpose_last(self._rot), self.lib)

    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SO3.exp(vec, a, LIB))
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return transpose_last(SO3.exp_integ(vec, a, LIB))
    
class SO3inertia(SO3):
    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
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
