from mathrobo.lie_abst import *
from mathrobo.basic import *

class SO3(LieAbstract):
  def __init__(self, r = identity(3), LIB = 'numpy'):
    '''
    Constructor
    '''
    self._rot = r
    self._lib = LIB
    
  def matrix(self):
    return self._rot
  
  def set_matrix(self, mat = identity(4)):
    self._rot = mat
    
  def inverse(self):
    return self._rot.transpose()

  def adjoint(self):
    return self._rot
  
  def set_adj_mat(self, mat = identity(3)):
    self._rot = mat

  def adj_inv(self):
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
  def mat(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if theta != 1.0:
        a_ = a*theta
      else:
        a_ = a
        
      if iszero(theta):
        return identity(3)
      else:
        x = vec[0]/theta
        y = vec[1]/theta
        z = vec[2]/theta           

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
  def integ_mat(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if theta != 1.0:
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

    mat[0,0] = k*sa + k*u*x*x
    mat[0,1] = k*u*x*y - k*v*z
    mat[0,2] = k*u*z*x + k*v*y
    mat[1,0] = k*u*x*y + k*v*z
    mat[1,1] = k*sa + k*u*y*y
    mat[1,2] = k*u*y*z - k*v*x
    mat[2,0] = k*u*z*x - k*v*y
    mat[2,1] = k*u*y*z + k*v*x
    mat[2,2] = k*sa + k*u*z*z

    return mat
  
  @staticmethod
  def integ2nd_mat(vec, a = 1., LIB = 'numpy'):
    """
      回転行列の積分の計算
      sympyの場合,vecの大きさは1を想定
    """
    if LIB == 'numpy':
      theta = norm(vec, LIB)
      if theta != 1.0:
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

    mat[0,0] = k*u  + k*w*x*x
    mat[0,1] = k*w*x*y - k*v*z
    mat[0,2] = k*w*z*x + k*v*y
    mat[1,0] = k*w*x*y + k*v*z
    mat[1,1] = k*u  + k*w*y*y
    mat[1,2] = k*w*y*z - k*v*x
    mat[2,0] = k*w*z*x - k*v*y
    mat[2,1] = k*w*y*z + k*v*x
    mat[2,2] = k*u  + k*w*z*z
    
    return mat
  
  @staticmethod
  def adj_hat(vec, LIB = 'numpy'):
    return SO3.hat(vec, LIB)
  
  @staticmethod
  def adj_hat_commute(vec, LIB = 'numpy'):
    return SO3.hat_commute(vec, LIB)
  
  @staticmethod
  def adj_mat(vec, a, LIB = 'numpy'):
    return SO3.mat(vec, a, LIB)
  
  @staticmethod
  def adj_integ_mat(vec, a, LIB = 'numpy'):
    return SO3.integ_mat(vec, a, LIB)
  
class SO3wre(SO3):
  @staticmethod
  def hat(vec, LIB = 'numpy'):
    return -SO3.hat(vec, LIB)
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    return SO3.hat(vec, LIB)
  
  @staticmethod
  def mat(vec, a, LIB = 'numpy'):
    return SO3.mat(vec, a, LIB).transpose()
  
  @staticmethod
  def integ_mat(vec, a, LIB = 'numpy'):
    return SO3.integ_mat(vec, a, LIB).transpose()
  
class SO3ine(SO3):
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
  def mat(vec, a, LIB = 'numpy'):
    return SO3.mat(vec, a, LIB).transpose()
  
  @staticmethod
  def integ_mat(vec, a, LIB = 'numpy'):
    return SO3.integ_mat(vec, a, LIB).transpose()
