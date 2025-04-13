from ..basic import *
from .lie_abst import *

class SO2(LieAbstract):
  _dof = 2
  def __init__(self, r = identity(2), LIB = 'numpy'):
    '''
    Constructor
    '''
    self._rot = r
    self._lib = LIB

  @staticmethod
  def dof():
    return 2
    
  def mat(self):
    return self._rot
  
  @staticmethod
  def set_mat(mat = identity(2)):
    return SO2(mat)
  
  def angle(self):
    _angle = np.acos((self._rot[0,0] + self._rot[1,1])*0.5) * 0.5 + np.asin((-self._rot[0,1] + self._rot[1,0])*0.5)*0.5
    return _angle
    
  @staticmethod
  def angle_to_rot_mat(a):
    m = np.array([
      [np.cos(a), -np.sin(a)],
      [np.sin(a), np.cos(a)]
    ]) 
    return m

  @staticmethod
  def set_quaternion(quaternion):
    return SO2(SO2.angle_to_rot_mat(quaternion))
    
  @staticmethod
  def eye():
    return SO2(identity(2))

  def inv(self):
    return self._rot.transpose()

  def mat_adj(self):
    return self._rot
  
  @staticmethod
  def set_mat_adj(mat = identity(3)):
    return SO2(mat)

  def inv_adj(self):
    return self._rot.transpose()

  def __matmul__(self, rval):
    if isinstance(rval, SO2):
      return SO2(self._rot @ rval._rot)
    elif isinstance(rval, np.ndarray):
      return self._rot @ rval
    else:
      TypeError("Right operand should be SO2 or numpy.ndarray")