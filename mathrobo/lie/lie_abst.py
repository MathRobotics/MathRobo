from ..basic import *

class LieAbstract:

  def __init__(self, LIB = 'numpy'): 
    '''
    Constructor
    '''
    pass

  @staticmethod
  def hat(vec, LIB = 'numpy'):
    '''
    hat operator on the tanget space vector
    '''
    pass
  
  @staticmethod
  def hat_commute(vec, LIB = 'numpy'):
    '''
    hat commute operator on the tanget space vector
    hat(a) @ b = hat_commute(b) @ a 
    '''
    pass

  @staticmethod
  def vee(vec_hat, LIB = 'numpy'):
    '''
    a = vee(hat(a))
    '''
    pass  
  
  @staticmethod
  def exp(vec, a, LIB = 'numpy'):
    pass

  @staticmethod
  def exp_integ(vec, a, LIB = 'numpy'):
    pass
  
  def inv(self):
    pass
  
  def mat_adj(self):
    '''
    adjoint expresion of Lie group
    '''
    pass
  
  @staticmethod
  def hat_adj(vec, LIB = 'numpy'):
    pass

  @staticmethod
  def hat_commute_adj(vec, LIB = 'numpy'):
    pass
  
  @staticmethod
  def exp_adj(vec, a, LIB = 'numpy'):
    pass
  
  @staticmethod
  def exp_integ_adj(vec, a, LIB = 'numpy'):
    pass