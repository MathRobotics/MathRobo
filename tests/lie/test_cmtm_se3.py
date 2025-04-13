import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_cmtm_se3():
  se3 = mr.SE3.rand()
  res = mr.CMTM[mr.SE3](se3)

  np.testing.assert_array_equal(res.mat(), se3.mat())
  
def test_cmtm_se3_vec1d():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(1,6) 

  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.zeros((8,8))
  mat[0:4,0:4] = mat[4:8,4:8] = se3.mat()
  mat[4:8,0:4] = se3.mat() @ se3.hat(vel[0])

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_se3_vec2d():
  se3 = mr.SE3.rand()  
  vec = np.random.rand(2,6)

  res = mr.CMTM[mr.SE3](se3, vec)
  
  mat = np.zeros((12,12))
  mat[0:4,0:4] = mat[4:8,4:8] = mat[8:12,8:12] = se3.mat()
  mat[4:8,0:4] = mat[8:12,4:8] = se3.mat() @ se3.hat(vec[0])
  mat[8:12,0:4] = se3.mat() @ (se3.hat(vec[1]) + se3.hat(vec[0]) @ se3.hat(vec[0])) * 0.5

  np.testing.assert_allclose(res.mat(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_se3_adj():
  se3 = mr.SE3.rand()
  res = mr.CMTM[mr.SE3](se3)

  np.testing.assert_allclose(res.mat_adj(), se3.mat_adj(), rtol=1e-15, atol=1e-15)
  
def test_cmtm_se3_vec1d():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(1,6) 

  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.zeros((12,12))
  mat[0:6,0:6] = mat[6:12,6:12] = se3.mat_adj()
  mat[6:12,0:6] = se3.mat_adj() @ se3.hat_adj(vel[0])

  np.testing.assert_array_equal(res.mat_adj(), mat)
  
def test_cmtm_se3_adj_vec2d():
  se3 = mr.SE3.rand()  
  vec = np.random.rand(2,6)

  res = mr.CMTM[mr.SE3](se3, vec)
  
  mat = np.zeros((18,18))
  mat[0:6,0:6] = mat[6:12,6:12] = mat[12:18,12:18] = se3.mat_adj()
  mat[6:12,0:6] = mat[12:18,6:12] = se3.mat_adj() @ se3.hat_adj(vec[0])
  mat[12:18,0:6] = se3.mat_adj() @ (se3.hat_adj(vec[1]) + se3.hat_adj(vec[0]) @ se3.hat_adj(vec[0])) * 0.5

  np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_se3_getter():
  se3 = mr.SE3.rand()  
  vec = np.random.rand(2,6)

  res = mr.CMTM[mr.SE3](se3,vec)
  
  np.testing.assert_array_equal(res.elem_mat(), se3.mat())
  np.testing.assert_array_equal(res.elem_vecs(0), vec[0])
  np.testing.assert_array_equal(res.elem_vecs(1), vec[1])
  
def test_cmtm_se3_inv():
  se3 = mr.SE3.rand()  

  for i in range(2):
    vel = np.random.rand(i,6)
    
    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(4*(i+1))

    np.testing.assert_allclose(res.mat() @ res.mat_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_inv_adj():
  se3 = mr.SE3.rand()   
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(6*(i+1))
    
    np.testing.assert_allclose(res.mat_adj() @ res.mat_inv_adj(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_tan_mat():
  se3 = mr.SE3.rand()   
  
  res = mr.CMTM[mr.SE3](se3)
  
  mat = np.eye(4)
  
  np.testing.assert_array_equal(res.tan_mat(), mat)
  
def test_cmtm_se3_vec1d_tan_mat():
  se3 = mr.SE3.rand()
  vel = np.random.rand(1,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(8)
  mat[4:8, 0:4] = - mr.SE3.hat(vel[0])
  
  np.testing.assert_array_equal(res.tan_mat(), mat)
  
def test_cmtm_se3_vec2d_tan_mat():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(2,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(12)
  mat[4:8, 0:4] = mat[8:12, 4:8] = - mr.SE3.hat(vel[0])
  mat[8:12, 0:4] = - (mr.SE3.hat(vel[1]) - mr.SE3.hat(vel[0]) @ mr.SE3.hat(vel[0])) 

  np.testing.assert_array_equal(res.tan_mat(), mat)
  
def test_cmtm_se3_tan_mat_adj():
  se3 = mr.SE3.rand() 
  
  res = mr.CMTM[mr.SE3](se3)
  
  mat = np.eye(6)
  
  np.testing.assert_array_equal(res.tan_mat_adj(), mat)
  
def test_cmtm_se3_vec1d_tan_mat_adj():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(1,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(12)
  mat[6:12, 0:6] = - mr.SE3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.tan_mat_adj(), mat)
  
def test_cmtm_se3_vec2d_tan_mat_adj():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(2,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(18)
  mat[6:12, 0:6] = mat[12:18, 6:12] = - mr.SE3.hat_adj(vel[0])
  mat[12:18, 0:6] = - (mr.SE3.hat_adj(vel[1]) - mr.SE3.hat_adj(vel[0]) @ mr.SE3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.tan_mat_adj(), mat)
  
def test_cmtm_se3_tan_inv():
  se3 = mr.SE3.rand()  
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(4*(i+1))

    np.testing.assert_allclose(res.tan_mat() @ res.tan_mat_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_tan_inv_adj():
  se3 = mr.SE3.rand() 
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(6*(i+1))

    np.testing.assert_allclose(res.tan_mat_adj() @ res.tan_mat_inv_adj(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_matmul():
  se3 = mr.SE3.rand()
  vel = np.random.rand(2,6)
  
  res1 = mr.CMTM[mr.SE3](se3, vel)
  res2 = mr.CMTM.eye(mr.SE3)
  
  mat1 = res1.mat() @ res2.mat()
  mat2 = mr.CMTM[mr.SE3](se3, vel).mat()
  
  np.testing.assert_allclose(mat1, mat2, rtol=1e-15, atol=1e-15)

def test_cmtm_se3_multiply():
  se3 = mr.SE3.rand()
  vel = np.random.rand(2,6)
  
  res1 = mr.CMTM[mr.SE3](se3, vel)
  res2 = mr.CMTM.eye(mr.SE3)
  
  mat1 = res1 @ res2
  mat2 = mr.CMTM[mr.SE3](se3, vel)
  
  np.testing.assert_allclose(mat1.mat(), mat2.mat(), rtol=1e-15, atol=1e-15)