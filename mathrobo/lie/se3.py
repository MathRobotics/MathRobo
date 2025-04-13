from ..basic import *
from .lie_abst import *
from .so3 import *

class SE3(LieAbstract):
  _dof = 6
  def __init__(self, rot = identity(3), pos = zeros(3), LIB = 'numpy'): 
    '''
    Constructor
    '''
    self._rot = rot
    self._pos = pos
    self.lib = LIB

  def dof(self):
    return self._dof
  
  def mat(self):
    mat = identity(4)
    mat[0:3,0:3] = self._rot
    mat[0:3,3] = self._pos
    return mat
  
  @staticmethod
  def set_mat(mat = identity(4)):
    return SE3(mat[0:3,0:3], mat[0:3,3])
  
  @staticmethod
  def set_pos_quaternion(pos, quaternion):
    return SE3(SO3.quaternion_to_rot_mat(quaternion), pos)

  def pos(self):
    return self._pos
  
  def rot(self):
    return self._rot
  
  def pos_quaternion(self):
    return self._pos, SO3.quaternion(SO3.set_mat(self._rot))

  @staticmethod
  def eye():
    return SE3(identity(3), zeros(3)) 

  def inv(self):
    return SE3(self._rot.transpose(), -self._rot.transpose() @ self._pos)

  def mat_inv(self):
    mat = identity(4, self.lib)
    mat[0:3,0:3] = self._rot.transpose()
    mat[0:3,3] = -self._rot.transpose() @ self._pos
    return mat
  
  def mat_adj(self):
    mat = zeros((6,6), self.lib)
    
    mat[0:3,0:3] = self._rot
    mat[3:6,0:3] = SO3.hat(self._pos, self.lib) @ self._rot
    mat[3:6,3:6] = self._rot
    
    return mat
  
  @staticmethod
  def set_mat_adj(mat = identity(6)):
    rot = (mat[0:3,0:3] + mat[3:6,3:6]) * 0.5
    pos = SO3.vee(mat[3:6,0:3] @ rot.transpose())
    
    return SE3(rot, pos)

  def mat_inv_adj(self):
    mat = zeros((6,6), self.lib)
    
    mat[0:3,0:3] = self._rot.transpose()
    mat[3:6,0:3] = -self._rot.transpose() @ SO3.hat(self._pos, self.lib)
    mat[3:6,3:6] = self._rot.transpose()
    
    return mat

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    '''
    hat operator on the tanget space vector
    '''
    mat = zeros((4,4), LIB)

    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[0:3,3] = vec[3:6]

    return mat
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    '''
    hat commute operator on the tanget space vector
    hat(a) @ b = hat_commute(b) @ a 
    '''
    mat = zeros((4,6), LIB)

    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    
    return -mat

  @staticmethod
  def vee(vec_hat, LIB = 'numpy'):
    '''
    a = vee(hat(a))
    '''
    vec = zeros(6, LIB)
    
    if(LIB == 'sympy'):
      vec[0:3,0] = SO3.vee(vec_hat[0:3,0:3], LIB)
      vec[3:6,0] = vec_hat[0:3,3]
    else:
      vec[0:3] = SO3.vee(vec_hat[0:3,0:3], LIB)
      vec[3:6] = vec_hat[0:3,3]

    return vec
  
  @staticmethod
  def exp(vec, a = 1., LIB = 'numpy'):
    '''
    同次変換行列の計算
    sympyの場合,vec[0:3]の大きさは1を想定
    '''
    if LIB == 'numpy':
      rot = vec[0:3]
      pos = vec[3:6]
    elif LIB == 'sympy':
      rot = sp.Matrix(vec[0:3])
      pos = sp.Matrix(vec[3:6])
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    mat = zeros((4,4), LIB)
    mat[0:3,0:3] = SO3.exp(rot, a, LIB)
    V = SO3.exp_integ(rot, a, LIB)

    mat[0:3,3] = V @ pos
    mat[3,3] = 1

    return mat
  
  @staticmethod
  def __integ_p_cross_r(vec, a = 1., LIB = 'numpy'):
    """
      p x Rの積分の計算
      sympyの場合,vec[0:3]の大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec[0:3], LIB)
      if not np.isclose(theta, 1.0):
        a_ = a*theta
      else:
        a_ = a

      if iszero(theta):
        return 0.5*a*a*SO3.hat(vec[3:6])
      else:
        u, v, w = vec[0:3] / theta
        x, y, z = vec[3:6]
        k = 1. / (theta*theta)
        
    elif LIB == 'sympy':
      a_ = a
      u, v, w, x, y, z = vec
      k = 1.
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)
    
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
  def exp_integ(vec, a = 1., LIB = 'numpy'):
    '''
    sympyの場合,vec[0:3]の大きさは1を想定
    '''
    if LIB == 'numpy':
      rot = vec[0:3]
      pos = vec[3:6]
    elif LIB == 'sympy':
      rot = sp.Matrix(vec[0:3])
      pos = sp.Matrix(vec[3:6])
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    mat = zeros((4,4), LIB)
    mat[0:3,0:3] = SO3.exp_integ(rot, a, LIB)
    V = SO3.exp_integ2nd(rot, a, LIB)

    mat[0:3,3] = V @ pos
    mat[3,3] = 1
    
    return mat
  
  @staticmethod
  def hat_adj(vec, LIB = 'numpy'):
    mat = zeros((6,6), LIB)

    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[3:6,3:6] = SO3.hat(vec[0:3], LIB)
    mat[3:6,0:3] = SO3.hat(vec[3:6], LIB)

    return mat
  
  @staticmethod
  def hat_commute_adj(vec, LIB = 'numpy'):
    return -SE3.hat_adj(vec, LIB)

  @staticmethod
  def vee_adj(vec_hat, LIB = 'numpy'):
    vec = zeros(6, LIB)
    
    vec[0,3] = 0.5*(SO3.vee(vec_hat[0:3,0:3], LIB) + SO3.vee(vec_hat[3:6,3:6]), LIB)
    vec[3,6] = SO3.vee(vec_hat[3:6,0:3], LIB)

    return vec
  
  @staticmethod
  def exp_adj(vec, a = 1., LIB = 'numpy'):
    '''
    SE3の随伴表現の計算
    sympyの場合,vec[0:3]の大きさは1を想定
    '''

    h = SE3.exp(vec, a, LIB = 'numpy')

    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = h[0:3,0:3]
    mat[3:6,0:3] = SO3.hat(h[0:3,3], LIB) @ h[0:3,0:3]
    mat[3:6,3:6] = h[0:3,0:3]

    return mat
  
  @staticmethod
  def exp_integ_adj(vec, a, LIB = 'numpy'):
    """
      SE3の随伴表現の積分の計算
      sympyの場合,vec[0:3]の大きさは1を想定
    """
    if LIB == 'numpy':
      rot = vec[0:3]
    elif LIB == 'sympy':
      rot = sp.Matrix(vec[0:3])
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")
    
    r = SO3.exp_integ(rot, a, LIB)

    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = r
    mat[3:6,0:3] = SE3.__integ_p_cross_r(vec, a, LIB)
    mat[3:6,3:6] = r

    return mat
  
  def __matmul__(self, rval):
    if isinstance(rval, SE3):
      return SE3(self._rot @ rval._rot, self._pos + self._rot @ rval._pos)
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
  def rand():
    p = np.random.rand(3) 
    return SE3(SO3.rand().mat(), p)
  
class SE3wrench(SE3):
  def mat(self):
    mat = zeros((6,6), self.lib)
    
    mat[0:3,0:3] = self._rot
    mat[0:3,3:6] = SO3.hat(self._pos, self.lib) @ self._rot
    mat[3:6,3:6] = self._rot
    
    return mat
  
  def mat_adj(self):
    mat = zeros((6,6), self.lib)
    
    mat[0:3,0:3] = self._rot
    mat[0:3,3:6] = SO3.hat(self._pos, self.lib) @ self._rot
    mat[3:6,3:6] = self._rot
    
    return mat

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[3:6,3:6] = SO3.hat(vec[0:3], LIB)
    mat[0:3,3:6] = SO3.hat(vec[3:6], LIB)

    return mat
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    mat = zeros((6,6), LIB)
    mat[0:3,0:3] = SO3.hat(vec[0:3], LIB)
    mat[0:3,3:6] = SO3.hat(vec[3:6], LIB)
    mat[3:6,0:3] = SO3.hat(vec[3:6], LIB)

    return -mat
  
  @staticmethod
  def exp(vec, a, LIB = 'numpy'):
    return SE3.exp_adj(vec, a, LIB).transpose()
  
  @staticmethod
  def exp_integ(vec, a, LIB = 'numpy'):
    return SE3.exp_integ_adj(vec, a, LIB).transpose()
  
'''
  Khalil, et al. 1995
'''
class SE3inertia(SE3):
  @staticmethod
  def hat(vec, LIB = 'numpy'):
    mat = np.zeros((6,6),LIB)

    mpg = vec[1:4]

    mat[0:3,0:3] = SE3inertia.hat(vec[4:10])
    mat[0:3,3:6] = SE3wrench.hat(mpg)
    mat[3:6,0:3] = SO3.hat(mpg)
    mat[3:6,3:6] = vec[0]*identity(3,LIB)

    return mat

  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    mat = zeros((6,10), LIB)

    v = vec[3:6]
    w = vec[0:3]

    mat[3:6,0] = v
    mat[0:3,1:4] = SE3wrench.hat_commute(v)
    mat[3:6,1:4] = SO3.hat_commute(w)
    mat[0:3,4:10] = SE3inertia.hat_commute(w)

    return mat