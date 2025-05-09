from ..basic import *
from .lie_abst import *

class SO3(LieAbstract):
  _dof = 3
  def __init__(self, r = identity(3), LIB = 'numpy'):
    '''
    Constructor
    '''
    self._rot = r
    self._lib = LIB

  @staticmethod
  def dof():
    return 3
    
  def mat(self):
    return self._rot
  
  @staticmethod
  def set_mat(mat = identity(3)):
    return SO3(mat)
  
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
  def quaternion_to_mat(quaternion):
    w, x, y, z = quaternion

    m = np.array([
      [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
      [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
      [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    
    return m

  @staticmethod
  def set_quaternion(quaternion):
    return SO3(SO3.quaternion_to_mat(quaternion))
  
  @staticmethod
  def mat_to_quaternion(mat):
    m = SO3(mat)
    return m.quaternion()
  
  @staticmethod
  def eye():
    return SO3(identity(3))

  def inv(self):
    return SO3(self._rot.transpose())
    
  def mat_inv(self):
    return self._rot.transpose()

  def mat_adj(self):
    return self._rot
  
  @staticmethod
  def set_mat_adj(mat = identity(3)):
    return SO3(mat)

  def mat_inv_adj(self):
    return self._rot.transpose()

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    mat = zeros((3,3), LIB)
    mat[1,2] = -vec[0]
    mat[2,1] =  vec[0]
    mat[2,0] = -vec[1]
    mat[0,2] =  vec[1]
    mat[0,1] = -vec[2]
    mat[1,0] =  vec[2]

    return mat
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    return -SO3.hat(vec, LIB)

  @staticmethod
  def vee(vec_hat, LIB = 'numpy'):
    vec = zeros(3, LIB)
    vec[0] = (-vec_hat[1,2] + vec_hat[2,1]) / 2
    vec[1] = (-vec_hat[2,0] + vec_hat[0,2]) / 2
    vec[2] = (-vec_hat[0,1] + vec_hat[1,0]) / 2
    return vec 

  @staticmethod
  def exp(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if not np.isclose(theta, 1.0):
        a_ = a*theta
      else:
        a_ = a
        
      if iszero(theta):
        return identity(3)
      else:
        x, y, z = vec / theta           

    elif LIB == 'sympy':
      a_ = a
      x = vec[0]
      y = vec[1]
      z = vec[2]
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)

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
  
  @staticmethod
  def exp_integ(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if not np.isclose(theta, 1.0):
        a_ = a*theta
      else:
        a_ = a

      if iszero(theta):
        return a*identity(3)
      else:
        x, y, z = vec/theta
        k = 1./theta
        
    elif LIB == 'sympy':
      a_ = a
      x, y, z = vec
      k = 1.
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)
    
    u = a_-sa
    v = (1-ca)

    mat[0,0] = k*(sa + u*x*x)
    mat[0,1] = k*(u*x*y - v*z)
    mat[0,2] = k*(u*z*x + v*y)
    mat[1,0] = k*(u*x*y + v*z)
    mat[1,1] = k*(sa + u*y*y)
    mat[1,2] = k*(u*y*z - v*x)
    mat[2,0] = k*(u*z*x - v*y)
    mat[2,1] = k*(u*y*z + v*x)
    mat[2,2] = k*(sa + u*z*z)

    return mat
  
  @staticmethod
  def exp_integ2nd(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if not np.isclose(theta, 1.0):
        a_ = a*theta
      else:
        a_ = a

      if iszero(theta):
        return a*identity(3)
      else:
        x, y, z = vec/theta
        k = 1./(theta*theta)
        
    elif LIB == 'sympy':
      a_ = a
      x, y, z = vec
      k = 1.
    else:
      raise ValueError("Unsupported library. Choose 'numpy' or 'sympy'.")

    sa = sin(a_, LIB)
    ca = cos(a_, LIB)

    mat = zeros((3,3), LIB)
    
    u = 1-ca
    v = a_-sa
    w = 0.5*a_**2-1+ca

    mat[0,0] = k*(u  + w*x*x)
    mat[0,1] = k*(w*x*y - v*z)
    mat[0,2] = k*(w*z*x + v*y)
    mat[1,0] = k*(w*x*y + v*z)
    mat[1,1] = k*(u  + w*y*y)
    mat[1,2] = k*(w*y*z - v*x)
    mat[2,0] = k*(w*z*x - v*y)
    mat[2,1] = k*(w*y*z + v*x)
    mat[2,2] = k*(u  + w*z*z)
    
    return mat
  
  @staticmethod
  def hat_adj(vec, LIB = 'numpy'):
    return SO3.hat(vec, LIB)
  
  @staticmethod
  def hat_commute_adj(vec, LIB = 'numpy'):
    return SO3.hat_commute(vec, LIB)
  
  @staticmethod
  def exp_adj(vec, a, LIB = 'numpy'):
    return SO3.exp(vec, a, LIB)
  
  @staticmethod
  def exp_integ_adj(vec, a, LIB = 'numpy'):
    return SO3.exp_integ(vec, a, LIB)

  @staticmethod
  def sub_tan_vec(val0, val1, type = 'bframe'):
    if type == 'bframe':
      vec = SO3.vee(val0.mat_inv() @ (val1._rot - val0._rot))
    elif type == 'fframe':
      vec = SO3.vee((val1._rot - val0._rot) @ val0.mat_inv())
    return vec

  def __matmul__(self, rval):
    if isinstance(rval, SO3):
      return SO3(self._rot @ rval._rot)
    elif isinstance(rval, np.ndarray):
      return self._rot @ rval
    else:
      TypeError("Right operand should be SO3 or numpy.ndarray")

  @staticmethod
  def rand():
    v = np.random.rand(3) 
    m = SO3.exp(v)
    return SO3(m)
  
class SO3wrench(SO3):
  @staticmethod
  def hat(vec, LIB = 'numpy'):
    return -SO3.hat(vec, LIB)
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    return SO3.hat(vec, LIB)
  
  @staticmethod
  def exp(vec, a, LIB = 'numpy'):
    return SO3.exp(vec, a, LIB).transpose()
  
  @staticmethod
  def exp_integ(vec, a, LIB = 'numpy'):
    return SO3.exp_integ(vec, a, LIB).transpose()
  
class SO3inertia(SO3):
  @staticmethod
  def hat(vec, LIB = 'numpy'):
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
  def hat_commute(vec, LIB = 'numpy'):
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
  def exp(vec, a, LIB = 'numpy'):
    return SO3.exp(vec, a, LIB).transpose()
  
  @staticmethod
  def exp_integ(vec, a, LIB = 'numpy'):
    return SO3.exp_integ(vec, a, LIB).transpose()
