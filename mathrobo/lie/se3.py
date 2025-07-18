from .lie_abst import *
from .so3 import *

from typing import Union, Tuple
import jax

class SE3(LieAbstract):
    _dof = 6
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
        if self.lib == 'jax':
            mat = jnp.block([
                [self._rot, self._pos[:, None]],
                [jnp.zeros((1, 3), dtype=self._rot.dtype), jnp.ones((1, 1), dtype=self._rot.dtype)]
            ])
            return mat
        elif self.lib == 'numpy':
            mat = np.eye(4, dtype=self._rot.dtype)
            mat[0:3, 0:3] = self._rot
            mat[0:3, 3] = self._pos
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
       
    
    @staticmethod
    def set_mat(mat = np.identity(4), LIB : str = 'numpy') -> 'SE3':
        return SE3(mat[0:3,0:3], mat[0:3,3], LIB)
    
    @staticmethod
    def set_pos_quaternion(pos: Union[np.ndarray, jnp.ndarray], 
                           quaternion: Union[np.ndarray, jnp.ndarray], LIB: str = 'numpy') -> 'SE3':
        return SE3(SO3.quaternion_to_mat(quaternion), pos, LIB)

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
        return SE3(self._rot.transpose(), -self._rot.transpose() @ self._pos, self.lib)

    def mat_inv(self) -> Union[np.ndarray, jnp.ndarray]:
        if self.lib == 'jax':
            mat = jnp.block([
                [self._rot.transpose() , -self._rot.transpose() @ self._pos],
                [jnp.zeros((1, 3), dtype=self._rot.dtype), jnp.ones((1, 1), dtype=self._rot.dtype)]
            ])
            return mat
        elif self.lib == 'numpy':
            mat = np.eye(4, dtype=self._rot.dtype)
            mat[0:3, 0:3] = self._rot.transpose()
            mat[0:3, 3] = -self._rot.transpose() @ self._pos
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        if self.lib == 'jax':
            mat = jnp.block([
                [self._rot, jnp.zeros((3, 3), dtype=self._rot.dtype)],
                [SO3.hat(self._pos, self.lib) @ self._rot, self._rot]
            ])
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros((6, 6), dtype=self._rot.dtype)
            mat[0:3, 0:3] = self._rot
            mat[3:6, 0:3] = SO3.hat(self._pos, self.lib) @ self._rot
            mat[3:6, 3:6] = self._rot
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def set_mat_adj(mat = np.identity(6), LIB : str = 'numpy') -> 'SE3':
        
        rot = (mat[0:3,0:3] + mat[3:6,3:6]) * 0.5
        pos = SO3.vee(mat[3:6,0:3] @ rot.transpose(), LIB)
        
        return SE3(rot, pos, LIB)

    def mat_inv_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        if self.lib == 'jax':
            mat = jnp.block([
                [self._rot.transpose(), jnp.zeros((3, 3), dtype=self._rot.dtype)],
                [-self._rot.transpose() @ SO3.hat(self._pos, self.lib), self._rot.transpose()]
            ])
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros((6, 6), dtype=self._rot.dtype)
            mat[0:3, 0:3] = self._rot.transpose()
            mat[3:6, 0:3] = -self._rot.transpose() @ SO3.hat(self._pos, self.lib)
            mat[3:6, 3:6] = self._rot.transpose()
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat operator on the tanget space vector
        '''
        if LIB == "jax":
            w, v = jnp.split(vec, 2, axis=-1)
            upper = jnp.concatenate(
                (SO3.hat(w, LIB), v.reshape(3, 1)), axis=1)  # (3,4)
            lower = jnp.zeros((1, 4), upper.dtype) # (1,4)

            return jnp.concatenate((upper, lower), axis=0)  # (4,4)
        elif LIB == 'numpy':
            mat = np.zeros((4,4))

            mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
            mat[0:3,3] = vec[3:6]

            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def hat_commute(vec: Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        hat commute operator on the tanget space vector
        hat(a) @ b = hat_commute(b) @ a 
        '''
        mat = np.zeros((4,6))

        mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
        
        return -mat

    @staticmethod
    def vee(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        a = vee(hat(a))
        '''
        if LIB == 'jax':
            w = SO3.vee(vec_hat[0:3,0:3], LIB)
            v = vec_hat[0:3,3]
            return jnp.concatenate((w, v))
        elif LIB == 'numpy':
            w = SO3.vee(vec_hat[0:3,0:3], LIB)
            v = vec_hat[0:3,3]
            return np.concatenate((w, v))
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'jax':
            rot, pos = jnp.split(vec, 2, axis=-1)
            R = SO3.exp(rot, a, LIB)
            V = SO3.exp_integ(rot, a, LIB)
            p = (V @ pos).reshape(3, 1)                # (3,1) ★ここで列化★
            return jnp.block([
                    [R,   p],
                    [jnp.zeros((1,3), dtype=vec.dtype), jnp.ones((1,1), dtype=vec.dtype)]
                ])
        elif LIB == 'numpy':
            rot, pos = vec[0:3], vec[3:6]

            mat = np.zeros((4,4))
            mat[0:3,0:3] = SO3.exp(rot, a, LIB)
            V = SO3.exp_integ(rot, a, LIB)

            mat[0:3,3] = V @ pos
            mat[3,3] = 1

            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")
    
    @staticmethod
    def __integ_p_cross_r(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """
            p x Rの積分の計算
        """
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
        sympyの場合,vec[0:3]の大きさは1を想定
        '''
        if LIB == 'numpy':
            rot = vec[0:3]
            pos = vec[3:6]
        else:
            raise ValueError("Unsupported library. Choose 'numpy'.")

        mat = np.zeros((4,4))
        mat[0:3,0:3] = SO3.exp_integ(rot, a, LIB)
        V = SO3.exp_integ2nd(rot, a, LIB)

        mat[0:3,3] = V @ pos
        mat[3,3] = 1
        
        return mat

    @staticmethod
    def hat_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:

        w, v = vec[:3], vec[3:]
        w_hat = SO3.hat(w, LIB)
        v_hat = SO3.hat(v, LIB)

        if LIB == 'jax':
            mat = jnp.block([
                [w_hat, jnp.zeros((3, 3), dtype=vec.dtype)],
                [v_hat, w_hat]
            ])
        elif LIB == 'numpy':
            mat = np.zeros((6, 6))
            mat[0:3, 0:3] = w_hat
            mat[3:6, 0:3] = v_hat
            mat[3:6, 3:6] = w_hat
        else:
            raise ValueError("Unsupported library. Choose 'numpy', 'jax'.")

        return mat
    
    @staticmethod
    def hat_commute_adj(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return -SE3.hat_adj(vec, LIB)

    @staticmethod
    def vee_adj(vec_hat : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'jax':
            w = 0.5 * ( SO3.vee(vec_hat[0:3,0:3], LIB) + SO3.vee(vec_hat[3:6,3:6], LIB) )
            v = SO3.vee(vec_hat[3:6,0:3], LIB)
            return jnp.concatenate((w, v))
        elif LIB == 'numpy':
            w = 0.5 * (SO3.vee(vec_hat[0:3, 0:3], LIB) + SO3.vee(vec_hat[3:6, 3:6], LIB))
            v = SO3.vee(vec_hat[3:6, 0:3], LIB)
            return np.concatenate([w, v])
        else:
            raise ValueError("Unsupported library. Choose 'numpy', 'jax'.")
    
    @staticmethod
    def exp_adj(vec : Union[np.ndarray, jnp.ndarray], a : float = 1., LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        '''
        SE3の随伴表現の計算
        sympyの場合,vec[0:3]の大きさは1を想定
        '''

        h = SE3.exp(vec, a, LIB)

        mat = zeros((6,6), LIB)
        mat[0:3,0:3] = h[0:3,0:3]
        mat[3:6,0:3] = SO3.hat(h[0:3,3], LIB) @ h[0:3,0:3]
        mat[3:6,3:6] = h[0:3,0:3]

        return mat
    
    @staticmethod
    def exp_integ_adj(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        """
            SE3の随伴表現の積分の計算
        """
        if LIB == 'numpy':
            rot = vec[0:3]
        elif LIB == 'jax':
            w, _ = jnp.split(vec, 2, axis=-1)
            n = jnp.linalg.norm(w)
            a_ = a * n
            ca = jnp.cos(a_)
            sa = jnp.sin(a_)
            A0 = jnp.eye(6) * a
            A1 = jnp.where(a_ == 0.0, 0.5*a_*a_, 0.5 * (4.0 - 4.0*ca - a_*sa)/ (n*n))
            A2 = jnp.where(a_ == 0.0, 0.0, 0.5 * (4.0*a_ - 5.0*sa + a_*ca)/ (n*n*n))
            A3 = jnp.where(a_ == 0.0, 0.0, 0.5 * (2.0 - 2.0*ca -a_*sa)/ (n*n*n*n))
            A4 = jnp.where(a_ == 0.0, 0.0, 0.5 * (2.0*a_ - 3*sa + a_*ca)/ (n*n*n*n*n))
            K = SE3.hat_adj(vec, 'jax')
            return A0 + A1*K + A2*K@K + A3*K@K@K + A4*K@K@K@K
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

        r = SO3.exp_integ(rot, a, LIB)

        mat = np.zeros((6,6))
        mat[0:3,0:3] = r
        mat[3:6,0:3] = SE3.__integ_p_cross_r(vec, a, LIB)
        mat[3:6,3:6] = r

        return mat

    @staticmethod
    def sub_tan_vec(val0 : 'SE3', val1 : 'SE3', 
                    type : str = 'bframe', LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:

        w = SO3.sub_tan_vec(SO3(val0.rot(),LIB), SO3(val1.rot(),LIB), type, LIB)

        if type == 'bframe':
            v = val0.rot().transpose() @ (val1.pos() - val0.pos())
        elif type == 'fframe':
            tmp = (val1.rot() - val0.rot()) @ val0.rot().transpose()
            v = (val1.pos() - val0.pos()) - tmp @ val0.pos()
        
        if LIB == 'numpy':
            vec = np.concatenate([w, v])
        elif LIB == 'jax':
            vec = jnp.concatenate([w, v])

        return vec

    def se3_mul(l_rot : Union[np.ndarray, jnp.ndarray], 
                l_pos : Union[np.ndarray, jnp.ndarray], 
                r_rot : Union[np.ndarray, jnp.ndarray], 
                r_pos : Union[np.ndarray, jnp.ndarray]) -> Tuple[Union[np.ndarray, jnp.ndarray], Union[np.ndarray, jnp.ndarray]]:
        return SO3.so3_mul(l_rot, r_rot), l_pos + l_rot @ r_pos
    
    def __matmul__(self, rval):
        if isinstance(rval, SE3):
            rot, pos = SE3.se3_mul(self._rot, self._pos, rval._rot, rval._pos)
            return SE3(rot, pos, self.lib)
        elif isinstance(rval, np.ndarray):
            if rval.shape[0] == 3:
                return self._rot @ rval + self._pos
            elif rval.shape == (6,):
                v = zeros(6)
                v[0:3] = self._rot @ rval[0:3]
                v[3:6] = SO3.hat(self._pos, self.lib) @ self._rot @ rval[0:3] + self._rot @ rval[3:6]
                return v
            elif rval.shape == (4,4):
                return self.mat() @ rval
            elif rval.shape == (6,6):
                return self.mat_adj() @ rval
        else:
            TypeError("Right operand should be SE3 or numpy.ndarray")

    @staticmethod
    def rand(LIB = 'numpy') -> 'SE3':
        if LIB == 'jax':
            p = jax.random.uniform(jax.random.PRNGKey(0), (3,))
        elif LIB == 'numpy':
            p = np.random.rand(3) 
        return SE3(SO3.rand(LIB).mat(), p, LIB)
    
class SE3wrench(SE3):
    def mat(self) -> Union[np.ndarray, jnp.ndarray]:
        if self.lib == 'jax':
            mat = jnp.zeros((6, 6), dtype=self._rot.dtype)
            mat = jnp.block([
                [self._rot, SO3.hat(self._pos, self.lib) @ self._rot],
                [jnp.zeros((3, 3), dtype=self._rot.dtype), self._rot]
            ])
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros((6,6))
            mat[0:3,0:3] = self._rot
            mat[0:3,3:6] = SO3.hat(self._pos) @ self._rot
            mat[3:6,3:6] = self._rot
            return mat
        else:
            raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")

    def mat_adj(self) -> Union[np.ndarray, jnp.ndarray]:
        if self.lib == 'jax':
            mat = jnp.zeros((6, 6), dtype=self._rot.dtype)
            mat = jnp.block([
                [self._rot, jnp.zeros((3, 3), dtype=self._rot.dtype)],
                [SO3.hat(self._pos, self.lib) @ self._rot, self._rot]
            ])
            return mat
        elif self.lib == 'numpy':
            mat = np.zeros((6,6))
            mat[0:3,0:3] = self._rot
            mat[0:3,3:6] = SO3.hat(self._pos, self.lib) @ self._rot
            mat[3:6,3:6] = self._rot
            return mat

    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        if LIB == 'jax':
            mat = jnp.zeros((6, 6), dtype=vec.dtype)
            mat = jnp.block([
                [SO3.hat(vec[0:3], LIB), vec[3:6].reshape(3, 1)],
                [jnp.zeros((3, 3), dtype=vec.dtype), SO3.hat(vec[3:6], LIB)]
            ])
            return mat
        elif LIB == 'numpy':
            mat = np.zeros((6, 6), dtype=vec.dtype)
            mat[0:3, 0:3] = SO3.hat(vec[0:3], LIB)
            mat[0:3, 3:6] = vec[3:6].reshape(3, 1)
            mat[3:6, 3:6] = SO3.hat(vec[3:6], LIB)
            return mat
    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        mat = np.zeros((6,6))
        mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
        mat[0:3,3:6] = SO3.hat(vec[3:6], LIB)
        mat[3:6,0:3] = SO3.hat(vec[3:6], LIB)

        return -mat
    
    @staticmethod
    def exp(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SE3.exp_adj(vec, a, LIB).transpose()
    
    @staticmethod
    def exp_integ(vec : Union[np.ndarray, jnp.ndarray], a : float, LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        return SE3.exp_integ_adj(vec, a, LIB).transpose()
    
'''
    Khalil, et al. 1995
'''
class SE3inertia(SE3):
    @staticmethod
    def hat(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        mat = np.zeros((6,6))

        mpg = vec[1:4]

        mat[0:3,0:3] = SE3inertia.hat(vec[4:10], LIB)
        mat[0:3,3:6] = SE3wrench.hat(mpg, LIB)
        mat[3:6,0:3] = SO3.hat(mpg, LIB)
        mat[3:6,3:6] = vec[0]*np.identity(3)

        return mat

    
    @staticmethod
    def hat_commute(vec : Union[np.ndarray, jnp.ndarray], LIB : str = 'numpy') -> Union[np.ndarray, jnp.ndarray]:
        mat = np.zeros((6,10))

        v = vec[3:6]
        w = vec[0:3]

        mat[3:6,0] = v
        mat[0:3,1:4] = SE3wrench.hat_commute(v, LIB)
        mat[3:6,1:4] = SO3.hat_commute(w, LIB)
        mat[0:3,4:10] = SE3inertia.hat_commute(w, LIB)

        return mat
