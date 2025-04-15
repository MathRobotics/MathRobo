from typing import TypeVar, Generic

from ..basic import *

T = TypeVar('T')

class CMTM(Generic[T]):
  def __init__(self, elem_mat, elem_vecs = np.array([]), LIB = 'numpy'): 
    '''
    Constructor
    '''
    self._mat = elem_mat
    self._vecs = elem_vecs
    self._dof = elem_mat.mat().shape[0]
    self._mat_size = elem_mat.mat().shape[0]
    self._mat_adj_size = elem_mat.mat_adj().shape[0]
    self._n = elem_vecs.shape[0] + 1
    self.lib = LIB
    
  def __mat_elem(self, p):
    if p == 0:
      return self._mat.mat()
    else:
      mat = zeros( (self._mat_size, self._mat_size) ) 
      for i in range(p):
        mat = mat + self.__mat_elem(p-(i+1)) @ self._mat.hat(self._vecs[i])

      return mat / p
    
  def mat(self):
    mat = identity(self._mat_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._mat_size*i:self._mat_size*(i+1),self._mat_size*j:self._mat_size*(j+1)] = self.__mat_elem(abs(i-j))
    return mat
  
  def __mat_adj_elem(self, p):
    if p == 0:
      return self._mat.mat_adj()
    else:
      mat = zeros( (self._mat_adj_size, self._mat_adj_size) ) 
      for i in range(p):
        mat = mat + self.__mat_adj_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i])
        
      return mat / p
    
  def mat_adj(self):
    mat = identity(self._mat_adj_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__mat_adj_elem(abs(i-j))
    return mat
  
  @staticmethod
  def eye(T):
    return CMTM(T.eye(), np.zeros((2,T.dof())))
  
  @staticmethod
  def rand(T):
    return CMTM(T.rand(), np.random.rand(2,T.dof()))  
  
  def elem_mat(self):
    return self._mat.mat()
  
  def elem_vecs(self, i):
    if(self._n - 1 > i ):
      if(self._vecs.ndim == 1):
        return self._vecs
      else:
        return self._vecs[i]
      
  def inv(self):
    vecs = np.zeros_like(self._vecs)
    for i in range(self._vecs.shape[0]):
      vecs[i] = self._mat @ self._vecs[i]
    return CMTM(self._mat.inv(), vecs)

  def __mat_inv_elem(self, p):
    if p == 0:
      return self._mat.mat_inv()
    else:
      mat = zeros( (self._mat_size, self._mat_size) ) 
      for i in range(p):
        mat = mat - self._mat.hat(self._vecs[i]) @ self.__mat_inv_elem(p-(i+1))
        
      return mat / p
  
  def mat_inv(self):
    mat = identity(self._mat_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._mat_size*i:self._mat_size*(i+1),self._mat_size*j:self._mat_size*(j+1)] = self.__mat_inv_elem(abs(i-j))
    return mat
  
  def __mat_inv_adj_elem(self, p):
    if p == 0:
      return self._mat.mat_inv_adj()
    else:
      mat = zeros( (self._mat_adj_size, self._mat_adj_size) ) 
      for i in range(p):
        mat = mat - self._mat.hat_adj(self._vecs[i]) @ self.__mat_inv_adj_elem(p-(i+1))
        
      return mat / p
  
  def mat_inv_adj(self):
    mat = identity(self._mat_adj_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__mat_inv_adj_elem(abs(i-j))
    return mat
  
  def __tan_mat_elem(self, p):
    if p == 0:
      return identity( self._mat_size ) 
    else:
      mat = zeros( (self._mat_size, self._mat_size) )
      for i in range(p):
        mat = mat - self.__tan_mat_elem(p-(i+1)) @ self._mat.hat(self._vecs[i])
      return mat
  
  def tan_mat(self):
    mat = identity(self._mat_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._mat_size*i:self._mat_size*(i+1),self._mat_size*j:self._mat_size*(j+1)] = self.__tan_mat_elem(abs(i-j))
    return mat
      
  def tan_mat_inv(self):
    mat = identity(self._mat_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._mat_size*i:self._mat_size*(i+1),self._mat_size*j:self._mat_size*(j+1)] = self._mat.hat(self._vecs[abs(i-j-1)])
    return mat
  
  def __tan_mat_adj_elem(self, p):
    if p == 0:
      return identity( self._mat_adj_size ) 
    else:
      mat = zeros( (self._mat_adj_size, self._mat_adj_size) )
      for i in range(p):
        mat = mat - self.__tan_mat_adj_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i])
      return mat
  
  def tan_mat_adj(self):
    mat = identity(self._mat_adj_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i >= j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__tan_mat_adj_elem(abs(i-j))
    return mat
  
  def tan_mat_inv_adj(self):
    mat = identity(self._mat_adj_size * self._n)
    for i in range(self._n):
      for j in range(self._n):
        if i > j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self._mat.hat_adj(self._vecs[abs(i-j-1)])
    return mat
  
  def __matmul__(self, rval):
    if isinstance(rval, CMTM):
      if self._n == rval._n:
        m = self._mat @ rval._mat
        v = np.zeros((self._n-1,self._mat.dof()))
        if self._n == 2:
          v[0] = rval._mat.mat_inv_adj() @ self._vecs[0] + rval._vecs[0]
        elif self._n == 3:
          v[0] = rval._mat.mat_inv_adj() @ self._vecs[0] + rval._vecs[0]
          v[1] = rval._mat.mat_inv_adj() @ self._vecs[1] + self._mat.hat_adj(rval._mat @ rval._vecs[0]) @ self._vecs[0] + rval._vecs[1]
        else:
          TypeError("Not supported n > 3")
        return CMTM(m, v)
      TypeError("Right operand should be same size in left operand")
    elif isinstance(rval, np.ndarray):
      return self.mat() @ rval
    else:
      TypeError("Right operand should be CMTM or numpy.ndarray")