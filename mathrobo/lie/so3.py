from .lie_abst import *

from typing import Union

import jax

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
        return SO3(SO3.quaternion_to_mat(quaternion), LIB)
    
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
        return SO3(self._rot.transpose(), self.lib)

    def mat_inv(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._rot.transpose()

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._rot
    
    @staticmethod
    def set_mat_adj(mat = np.identity(3), LIB : str = 'numpy') -> 'SO3':
        return SO3(mat, LIB)

    def mat_inv_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._rot.transpose()

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == "jax":
            if vec.ndim == 1:
                vx, vy, vz = vec
                return jnp.array([
                                [   0, -vz,  vy],
                                [  vz,   0, -vx],
                                [ -vy,  vx,   0]], dtype=vec.dtype)
            vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
            zero = jnp.zeros_like(vx)
            return jnp.stack((zero, -vz, vy, vz, zero, -vx, -vy, vx, zero), axis=-1).reshape(vec.shape[:-1] + (3, 3))
        elif LIB == "numpy":
            if vec.ndim == 1:
                vx, vy, vz = vec
                return np.array([
                                [   0, -vz,  vy],
                                [  vz,   0, -vx],
                                [ -vy,  vx,   0]], dtype=vec.dtype)
            vx, vy, vz = vec[..., 0], vec[..., 1], vec[..., 2]
            zero = np.zeros_like(vx)
            return np.stack((zero, -vz, vy, vz, zero, -vx, -vy, vx, zero), axis=-1).reshape(vec.shape[:-1] + (3, 3))
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SO3.hat(vec, LIB)

    @staticmethod
    def vee(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'jax':
            return jnp.array([
                0.5 * (vec_hat[2, 1] - vec_hat[1, 2]),
                0.5 * (vec_hat[0, 2] - vec_hat[2, 0]),
                0.5 * (vec_hat[1, 0] - vec_hat[0, 1])
            ], dtype=vec_hat.dtype)
        elif LIB == 'numpy':
            return np.array([
                0.5 * (vec_hat[2, 1] - vec_hat[1, 2]),
                0.5 * (vec_hat[0, 2] - vec_hat[2, 0]),
                0.5 * (vec_hat[1, 0] - vec_hat[0, 1])
            ], dtype=vec_hat.dtype)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'numpy':
            theta = np.linalg.norm(vec)
            if not math.isclose(theta, 1.0):
                a_ = a*theta
            else:
                a_ = a

            if math.isclose(theta, 0):
                return np.identity(3)
            else:
                x, y, z = vec / theta

            sa = np.sin(a_)
            ca = np.cos(a_)

            mat = np.zeros((3,3))

            mat[0,0] = ca + (1-ca)*x*x
            mat[0,1] = (1-ca)*x*y - sa*z
            mat[0,2] = (1-ca)*x*z + sa*y
            mat[1,0] = (1-ca)*y*x + sa*z
            mat[1,1] = ca + (1-ca)*y*y
            mat[1,2] = (1-ca)*y*z - sa*x
            mat[2,0] = (1-ca)*z*x - sa*y
            mat[2,1] = (1-ca)*z*y + sa*x
            mat[2,2] = ca + (1-ca)*z*z

            return mat
        elif LIB == 'jax':
            n = jnp.linalg.norm(vec)
            a_  = n * a
            I  = jnp.eye(3, dtype=vec.dtype)
            ca = jnp.cos(a_)
            sa = jnp.sin(a_)

            A  = jnp.where(a_ == 0.0, 0.0, sa)
            B  = jnp.where(a_ == 0.0, 0.0, (1.0 - ca))

            K = jnp.where(n == 0.0, jnp.zeros((3,3)), SO3.hat(vec/n, 'jax'))

            return I + A * K + B * (K @ K)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'numpy':
            if vec.ndim > 1:
                flat = vec.reshape((-1, 3))
                return np.stack([SO3.exp_integ(v, a, LIB) for v in flat]).reshape(vec.shape[:-1] + (3, 3))
            theta = np.linalg.norm(vec)
            if abs(a) * theta < 1e-12:
                return a * np.identity(3)

            a_ = a * theta
            x, y, z = vec / theta
            k = 1. / theta
            if abs(a_) < 1e-3:
                x2 = a_ * a_
                sa = a_ * (1 - x2/6 + x2*x2/120 - x2*x2*x2/5040)
                v = x2 * (1/2 - x2/24 + x2*x2/720 - x2*x2*x2/40320)
            else:
                sa = np.sin(a_)
                v = 1 - np.cos(a_)
            u = a_ - sa

            return k * np.array([
                [sa + u*x*x, u*x*y - v*z, u*x*z + v*y],
                [u*x*y + v*z, sa + u*y*y, u*y*z - v*x],
                [u*x*z - v*y, u*y*z + v*x, sa + u*z*z],
            ])
        elif LIB == 'jax':
            vec = jnp.asarray(vec)
            theta2 = jnp.sum(vec * vec, axis=-1)
            tiny = (a * a) * theta2 < 1e-24
            safe_theta2 = jnp.where(tiny, jnp.ones_like(theta2), theta2)
            theta = jnp.sqrt(safe_theta2)
            a_ = a * theta
            x2 = a_ * a_
            series = x2 < 1e-6
            closed_theta2 = jnp.where(series, jnp.ones_like(theta2), safe_theta2)
            closed_theta = jnp.sqrt(closed_theta2)
            closed_angle = a * closed_theta
            A_closed = (1.0 - jnp.cos(closed_angle)) / closed_theta2
            B_closed = (closed_angle - jnp.sin(closed_angle)) / (closed_theta2 * closed_theta)
            A_series = a*a * (1/2 - x2/24 + x2*x2/720 - x2*x2*x2/40320)
            B_series = a*a*a * (1/6 - x2/120 + x2*x2/5040 - x2*x2*x2/362880)
            A = jnp.where(series, A_series, A_closed)
            B = jnp.where(series, B_series, B_closed)
            K = SO3.hat(vec, 'jax')
            result = a * jnp.eye(3, dtype=vec.dtype) + A[..., None, None] * K + B[..., None, None] * (K @ K)
            return jnp.where(tiny[..., None, None], a * jnp.eye(3, dtype=vec.dtype), result)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def exp_integ2nd(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'numpy':
            if vec.ndim > 1:
                flat = vec.reshape((-1, 3))
                return np.stack([SO3.exp_integ2nd(v, a, LIB) for v in flat]).reshape(vec.shape[:-1] + (3, 3))
            theta = np.linalg.norm(vec)
            if abs(a) * theta < 1e-12:
                return 0.5 * a * a * np.identity(3)

            a_ = a * theta
            x, y, z = vec / theta
            k = 1. / (theta * theta)
            if abs(a_) < 1e-3:
                x2 = a_ * a_
                u = x2 * (1/2 - x2/24 + x2*x2/720 - x2*x2*x2/40320)
                v = a_ * x2 * (1/6 - x2/120 + x2*x2/5040 - x2*x2*x2/362880)
                w = x2*x2 * (1/24 - x2/720 + x2*x2/40320 - x2*x2*x2/3628800)
            else:
                sa = np.sin(a_)
                ca = np.cos(a_)
                u = 1 - ca
                v = a_ - sa
                w = 0.5 * a_**2 - 1 + ca

            return k * np.array([
                [u + w*x*x, w*x*y - v*z, w*x*z + v*y],
                [w*x*y + v*z, u + w*y*y, w*y*z - v*x],
                [w*x*z - v*y, w*y*z + v*x, u + w*z*z],
            ])
        elif LIB == 'jax':
            vec = jnp.asarray(vec)
            theta2 = jnp.sum(vec * vec, axis=-1)
            tiny = (a * a) * theta2 < 1e-24
            safe_theta2 = jnp.where(tiny, jnp.ones_like(theta2), theta2)
            theta = jnp.sqrt(safe_theta2)
            a_ = a * theta
            x2 = a_ * a_
            series = x2 < 1e-6
            closed_theta2 = jnp.where(series, jnp.ones_like(theta2), safe_theta2)
            closed_theta = jnp.sqrt(closed_theta2)
            closed_angle = a * closed_theta
            A_closed = (closed_angle - jnp.sin(closed_angle)) / (closed_theta2 * closed_theta)
            B_closed = (0.5 * closed_angle * closed_angle - 1.0 + jnp.cos(closed_angle)) / (closed_theta2 * closed_theta2)
            A_series = a*a*a * (1/6 - x2/120 + x2*x2/5040 - x2*x2*x2/362880)
            B_series = a**4 * (1/24 - x2/720 + x2*x2/40320 - x2*x2*x2/3628800)
            A = jnp.where(series, A_series, A_closed)
            B = jnp.where(series, B_series, B_closed)
            K = SO3.hat(vec, 'jax')
            result = 0.5 * a * a * jnp.eye(3, dtype=vec.dtype) + A[..., None, None] * K + B[..., None, None] * (K @ K)
            return jnp.where(tiny[..., None, None], 0.5 * a * a * jnp.eye(3, dtype=vec.dtype), result)
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

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
        elif isinstance(rval, np.ndarray):
            return SO3.so3_mul(self._rot, rval)
        else:
            TypeError("Right operand should be SO3 or numpy.ndarray")

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
        return SO3wrench(self._rot.transpose(), self.lib)

    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.exp(vec, a, LIB).transpose()
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.exp_integ(vec, a, LIB).transpose()
    
class SO3inertia(SO3):
    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        mat = zeros((3,3), LIB)

        mat[0,0] = vec[0]
        mat[0,1] = vec[5]
        mat[0,2] = vec[4]
        mat[1,0] = vec[5]
        mat[1,1] = vec[1]
        mat[1,2] = vec[3]
        mat[2,0] = vec[4]
        mat[2,1] = vec[3]
        mat[2,2] = vec[2]

        return mat
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        mat = zeros((3, 6), LIB)

        mat[0,0] = vec[0]
        mat[1,1] = vec[1]
        mat[2,2] = vec[2]

        mat[1,5] = vec[0]
        mat[2,4] = vec[0]
        mat[2,3] = vec[1]
        mat[0,5] = vec[1]
        mat[0,4] = vec[2]
        mat[1,3] = vec[2]

        return mat
    
    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.exp(vec, a, LIB).transpose()
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SO3.exp_integ(vec, a, LIB).transpose()
