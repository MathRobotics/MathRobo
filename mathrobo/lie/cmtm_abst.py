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

  def __check_output_order(self, output_order):
    if output_order is None:
      output_order = self._n
    if output_order > self._n:
      raise TypeError("Output order should be less than or equal to the order of CMTM")
    return output_order
    
  def __mat_elem(self, p):
    if p == 0:
      return self._mat.mat()
    else:
      mat = zeros( (self._mat_size, self._mat_size) ) 
      for i in range(p):
        mat = mat + self.__mat_elem(p-(i+1)) @ self._mat.hat(self._vecs[i])

      return mat / p
    
  def mat(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    
    mat = identity(self._mat_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
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
    
  def mat_adj(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    
    mat = identity(self._mat_adj_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
        if i >= j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__mat_adj_elem(abs(i-j))
    return mat

  @staticmethod
  def set_mat(T, mat : np.ndarray):
    size = T.eye().mat().shape[0]
    if mat.shape[0] % size != 0:
      raise TypeError("Matrix size is not same as element matrix")
   
    n = int(mat.shape[0] / size)

    tmp = np.zeros((n, size, size))
    for i in range(n):
      for j in range(n-i):
        tmp[i] += mat[(j+i)*size:(j+i+1)*size, j*size:(j+1)*size]
      tmp[i] = tmp[i] / (n-i)
      
    m = T.set_mat(tmp[0])
    vs = np.zeros((n-1, T.dof()))
    for i in range(n-1):
      m_tmp = np.zeros((size, size))
      for j in range(i):
        m_tmp += tmp[i-j] @ T.hat(vs[j])
      vs[i] = T.vee( m.inv() @  ( tmp[i+1] * (i+1) - m_tmp) )

    return CMTM(m, vs)
  
  @staticmethod
  def eye(T, order = 3):
    return CMTM(T.eye(), np.zeros((order-1,T.dof())))
  
  @staticmethod
  def rand(T, order = 3):
    return CMTM(T.rand(), np.random.rand(order-1,T.dof()))  
  
  def elem_mat(self):
    return self._mat.mat()
  
  def elem_vecs(self, i):
    if(self._n - 1 > i ):
      if(self._vecs.ndim == 1):
        return self._vecs
      else:
        return self._vecs[i]
      
  def vecs(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    return self._vecs[:output_order]
  
  def vecs_flat(self, output_order = None):
    output_order = self.__check_output_order(output_order)

    return self._vecs[:output_order].flatten()
  
  def tan_vecs(self, output_order = None):
    output_order = self.__check_output_order(output_order)
  
      
  def inv(self):
    vecs = np.zeros_like(self._vecs)
    if self._n < 4:
      for i in range(self._n-1):
        vecs[i] = -self._mat.mat_adj() @ self._vecs[i]
      return CMTM(self._mat.inv(), vecs)
    else:
      # tentative implementation
      return CMTM.set_mat(type(self._mat), self.mat_inv())

  def __mat_inv_elem(self, p):
    if p == 0:
      return self._mat.mat_inv()
    else:
      mat = zeros( (self._mat_size, self._mat_size) ) 
      for i in range(p):
        mat = mat - self._mat.hat(self._vecs[i]) @ self.__mat_inv_elem(p-(i+1))
        
      return mat / p
  
  def mat_inv(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    
    mat = identity(self._mat_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
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
  
  def mat_inv_adj(self, output_order = None):
    output_order = self.__check_output_order(output_order)

    mat = identity(self._mat_adj_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
        if i >= j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__mat_inv_adj_elem(abs(i-j))
    return mat
  
  @staticmethod
  def __hat_func(hat, vecs):
    n = vecs.shape[0]
    m = hat(vecs[0]).shape[0]
    mat = np.zeros((m*n,m*n))
    for i in range(n):
      tmp = hat(vecs[i])
      for j in range(n-i):
        mat[m*i+m*j:m*(i+1)+m*j,m*j:m*(j+1)] = tmp
    return mat
  
  @staticmethod
  def hat(T, vecs):
    return CMTM.__hat_func(T.hat, vecs)

  @staticmethod
  def hat_adj(T, vecs):
    return CMTM.__hat_func(T.hat_adj, vecs)

  @staticmethod
  def __vee_func(dof, vee, mat):
    '''
    dof : dof of lie group
    vee : vee function
    '''
    n = vee(np.zeros((dof,dof))).shape[0]
    m = int(mat.shape[0] / n)
    vecs = np.zeros((m,n))
    for i in range(m):
      tmp = np.zeros(n)
      for j in range(m-i):
        tmp += vee( mat[(j+i)*n:(j+i+1)*n, j*n:(j+1)*n] )
      vecs[i] = tmp / (m-i)
    return vecs

  @staticmethod
  def vee(T, hat_mat):
    return CMTM.__vee_func(T.dof(), T.vee, hat_mat)
  
  @staticmethod
  def vee_adj(T, hat_mat):
    return CMTM.__vee_func(T.dof(), T.vee_adj, hat_mat)
  
  def __ptan_map_elem(self, p):
    if p == 0:
      return identity( self._mat_size ) 
    else:
      mat = zeros( (self._mat_size, self._mat_size) )
      for i in range(p):
        mat = mat - self.__ptan_map_elem(p-(i+1)) @ self._mat.hat(self._vecs[i])
      return mat
  
  def ptan_map(self, output_order = None):
    output_order = self.__check_output_order(output_order)

    mat = identity(self._mat_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
        if i >= j :
          mat[self._mat_size*i:self._mat_size*(i+1),self._mat_size*j:self._mat_size*(j+1)] = self.__ptan_map_elem(abs(i-j))
    return mat
      
  def ptan_map_inv(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    mat = identity(self._mat_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
        if i > j :
          mat[self._mat_size*i:self._mat_size*(i+1),self._mat_size*j:self._mat_size*(j+1)] = self._mat.hat(self._vecs[abs(i-j-1)])
    return mat
  
  def __ptan_map_adj_elem(self, p):
    if p == 0:
      return identity( self._mat_adj_size ) 
    else:
      mat = zeros( (self._mat_adj_size, self._mat_adj_size) )
      for i in range(p):
        mat = mat - self.__ptan_map_adj_elem(p-(i+1)) @ self._mat.hat_adj(self._vecs[i])
      return mat
  
  def ptan_map_adj(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    mat = identity(self._mat_adj_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
        if i >= j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self.__ptan_map_adj_elem(abs(i-j))
    return mat
  
  def ptan_map_inv_adj(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    mat = identity(self._mat_adj_size * output_order)
    for i in range(output_order):
      for j in range(output_order):
        if i > j :
          mat[self._mat_adj_size*i:self._mat_adj_size*(i+1),self._mat_adj_size*j:self._mat_adj_size*(j+1)] = self._mat.hat_adj(self._vecs[abs(i-j-1)])
    return mat

  @staticmethod
  def ptan_to_tan(dof, output_order):
    '''
    Convert matrix the pseudo tangent vector to tangent vector.
    '''
    k = 1
    mat = np.zeros((dof*output_order, dof*output_order))
    for i in range(output_order):
      mat[i*dof:(i+1)*dof,i*dof:(i+1)*dof] = identity(dof) / k
      k = k * (i + 1)
    return mat

  @staticmethod
  def tan_to_ptan(dof, output_order):
    '''
    Convert the tangent vector to pseudo tangent vector.
    '''
    k = 1
    mat = np.zeros((dof*output_order, dof*output_order))
    for i in range(output_order):
      mat[i*dof:(i+1)*dof,i*dof:(i+1)*dof] = identity(dof) * k
      k = k * (i + 1)
    return mat

  def tan_map(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    return self.ptan_to_tan(self._mat_size, output_order) @ self.ptan_map(output_order)

  def tan_map_inv(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    return self.ptan_map_inv(output_order) @ self.tan_to_ptan(self._mat_size, output_order)

  def tan_map_adj(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    return self.ptan_to_tan(self._mat_adj_size, output_order) @ self.ptan_map_adj(output_order)

  def tan_map_inv_adj(self, output_order = None):
    output_order = self.__check_output_order(output_order)
    return self.ptan_map_inv_adj(output_order) @ self.tan_to_ptan(self._mat_adj_size, output_order)

  @staticmethod
  def sub_vec(lval, rval, type = 'bframe') -> np.ndarray: 
    if lval._n != rval._n:
      raise TypeError("Left operand should be same order in right operand")
    if lval._dof != rval._dof:
      raise TypeError("Left operand should be same dof in right operand")

    dof = lval._mat._dof
    vec = np.zeros((lval._n * dof))
    vec[:dof] = lval._mat.sub_tan_vec(lval._mat, rval._mat, type)

    for i in range(1,lval._n):
      vec[dof*i:dof*(i+1)] = rval._vecs[i-1] - lval._vecs[i-1]

    return vec
  
  @staticmethod
  def sub_ptan_vec(lval, rval, type = 'bframe') -> np.ndarray: 
    '''
    Subtract the psuedu tangent vector of two CMTM objects.
    '''
    if lval._n != rval._n:
      raise TypeError("Left operand should be same order in right operand")
    if lval._dof != rval._dof:
      raise TypeError("Left operand should be same dof in right operand")

    dof = lval._mat._dof
    vec = np.zeros((lval._n * dof))
    vec[:dof] = lval._mat.sub_tan_vec(lval._mat, rval._mat, type)
    for i in range(lval._n-1):
      vec[dof*(i+1):dof*(i+2)] = (rval._vecs[i] - lval._vecs[i])
      for j in range(i+1):
        vec[dof*(i+1):dof*(i+2)] += (lval._mat.hat_adj(vec[dof*j:dof*(j+1)]) @ lval._vecs[i-j])

    return vec

  @staticmethod
  def sub_tan_vec(lval, rval, type = 'bframe') -> np.ndarray:
    if lval._n != rval._n:
      raise TypeError("Left operand should be same order in right operand")
    if lval._dof != rval._dof:
      raise TypeError("Left operand should be same dof in right operand")

    if type == 'bframe':
      vec = lval.mat_inv() @ (rval.mat() - lval.mat())
    elif type == 'fframe':
      vec = (rval.mat() - lval.mat()) @ lval.mat_inv()
    
    return vec

  def __matmul__(self, rval):
    if isinstance(rval, CMTM):
      if self._n == rval._n:
        if self._n > 3:
          # tentative implementation
          return CMTM.set_mat(type(self._mat), self.mat() @ rval.mat())
        m = self._mat @ rval._mat
        v = np.zeros((self._n-1,self._mat.dof()))
        if self._n > 1:
          v[0] = rval._mat.mat_inv_adj() @ self._vecs[0] + rval._vecs[0]
        if self._n > 2:
          v[1] = rval._mat.mat_inv_adj() @ self._vecs[1] + self._mat.hat_adj(rval._mat.mat_inv_adj() @ self._vecs[0]) @ rval._vecs[0] + rval._vecs[1]
        return CMTM(m, v)
      else:
        TypeError("Right operand should be same size in left operand")
    elif isinstance(rval, np.ndarray):
      return self.mat() @ rval
    else:
      TypeError("Right operand should be CMTM or numpy.ndarray")

  def print(self):
    print("mat")
    print(self._mat.mat())
    print("vecs")
    print(self._vecs)