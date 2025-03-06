import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_cmtm_so3():
  v = np.random.rand(3) 
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)
  res = mr.CMTM[mr.SO3](so3)

  np.testing.assert_array_equal(res.mat(), so3.mat())
  
def test_cmtm_so3_vec1d():
  v = np.random.rand(3) 
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)  
  vel = np.random.rand(1,3) 

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = so3.mat()
  mat[3:6,0:3] = so3.mat() @ so3.hat(vel[0])

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_so3_vec2d():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vec = np.random.rand(2,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  
  mat = np.zeros((9,9))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = so3.mat()
  mat[3:6,0:3] = mat[6:9,3:6] = so3.mat() @ so3.hat(vec[0])
  mat[6:9,0:3] = so3.mat() @ (so3.hat(vec[1]) + so3.hat(vec[0]) @ so3.hat(vec[0])) * 0.5

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_so3_adj():
  v = np.random.rand(3) 
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)
  res = mr.CMTM[mr.SO3](so3)

  np.testing.assert_array_equal(res.mat_adj(), so3.mat_adj())
  
def test_cmtm_so3_vec1d():
  v = np.random.rand(3) 
  r = mr.SO3.exp(v)

  so3 = mr.SO3(r)  
  vel = np.random.rand(1,3) 

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = so3.mat_adj()
  mat[3:6,0:3] = so3.mat_adj() @ so3.hat_adj(vel[0])

  np.testing.assert_array_equal(res.mat_adj(), mat)
  
def test_cmtm_so3_adj_vec2d():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vec = np.random.rand(2,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  
  mat = np.zeros((9,9))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = so3.mat_adj()
  mat[3:6,0:3] = mat[6:9,3:6] = so3.mat_adj() @ so3.hat_adj(vec[0])
  mat[6:9,0:3] = so3.mat_adj() @ (so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) * 0.5

  np.testing.assert_array_equal(res.mat_adj(), mat)
  
def test_cmtm_so3_getter():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vec = np.random.rand(2,3)

  res = mr.CMTM[mr.SO3](so3,vec)
  
  np.testing.assert_array_equal(res.elem_mat(), so3.mat())
  np.testing.assert_array_equal(res.elem_vecs(0), vec[0])
  np.testing.assert_array_equal(res.elem_vecs(1), vec[1])
  
def test_cmtm_so3_inv():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  
  for i in range(3):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.mat() @ res.inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_so3_inv_adj():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  
  for i in range(3):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.mat_adj() @ res.inv_adj(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_so3_tangent_mat():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  
  res = mr.CMTM[mr.SO3](so3)
  
  mat = np.eye(3)
  
  np.testing.assert_array_equal(res.tangent_mat(), mat)
  
def test_cmtm_so3_vec1d_tangent_mat():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat(vel[0])
  
  np.testing.assert_array_equal(res.tangent_mat(), mat)
  
def test_cmtm_so3_vec2d_tangent_mat():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat(vel[0])
  mat[6:9, 0:3] = - (mr.SO3.hat(vel[1]) - mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0])) 

  np.testing.assert_array_equal(res.tangent_mat(), mat)
  
def test_cmtm_so3_tangent_mat_adj():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  
  res = mr.CMTM[mr.SO3](so3)
  
  mat = np.eye(3)
  
  np.testing.assert_array_equal(res.tangent_mat_adj(), mat)
  
def test_cmtm_so3_vec1d_tangent_mat_adj():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.tangent_mat_adj(), mat)
  
def test_cmtm_so3_vec2d_tangent_mat_adj():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 0:3] = - (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.tangent_mat_adj(), mat)
  
def test_cmtm_so3_tangent_inv():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  
  for i in range(3):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.tangent_mat() @ res.tangent_mat_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_so3_tangent_inv_adj():
  v = np.zeros(3)
  r = mr.SO3.exp(v)
  so3 = mr.SO3(r)  
  
  for i in range(3):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.tangent_mat_adj() @ res.tangent_mat_inv_adj(), mat, rtol=1e-15, atol=1e-15)