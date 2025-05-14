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

def test_cmtm_so3_inv():
  se3 = mr.SE3.rand()
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)

    expected_mat = np.eye(4*(i+1))
    result_mat = res @ res.inv()
    
    np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_se3_mat_inv():
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

def test_cmtm_se3_hat():
  vec = np.random.rand(1,6)

  res = mr.CMTM.hat(mr.SE3, vec)
  mat = mr.SE3.hat(vec[0])

  np.testing.assert_array_equal(res, mat)

def test_cmtm_se3_hat2():
  vec = np.random.rand(2,6)

  res = mr.CMTM.hat(mr.SE3, vec)
  mat = np.zeros((8,8))
  mat[0:4,0:4] = mat[4:8,4:8] = mr.SE3.hat(vec[0])
  mat[4:8,0:4] = mr.SE3.hat(vec[1])  

  np.testing.assert_array_equal(res, mat)

def test_cmtm_se3_hat_adj():
  vec = np.random.rand(1,6)

  res = mr.CMTM.hat_adj(mr.SE3, vec)
  mat = mr.SE3.hat_adj(vec[0])

  np.testing.assert_array_equal(res, mat)

def test_cmtm_se3_hat_adj2():
  vec = np.random.rand(2,6)

  res = mr.CMTM.hat_adj(mr.SE3, vec)
  mat = np.zeros((12,12))
  mat[0:6,0:6] = mat[6:12,6:12] = mr.SE3.hat_adj(vec[0])
  mat[6:12,0:6] = mr.SE3.hat_adj(vec[1])  

  np.testing.assert_array_equal(res, mat)

def test_cmtm_se3_hat_adj3():
  vec = np.random.rand(3,6)

  res = mr.CMTM.hat_adj(mr.SE3, vec)
  mat = np.zeros((18,18))
  mat[0:6,0:6] = mat[6:12,6:12] = mat[12:18,12:18] = mr.SE3.hat_adj(vec[0])
  mat[6:12,0:6] = mat[12:18,6:12] = mr.SE3.hat_adj(vec[1])
  mat[12:18,0:6] = mr.SE3.hat_adj(vec[2])  

  np.testing.assert_array_equal(res, mat)
    
def test_cmtm_se3_tan_map():
  se3 = mr.SE3.rand()   
  
  res = mr.CMTM[mr.SE3](se3)
  
  mat = np.eye(4)
  
  np.testing.assert_array_equal(res.tan_map(), mat)
  
def test_cmtm_se3_vec1d_tan_map():
  se3 = mr.SE3.rand()
  vel = np.random.rand(1,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(8)
  mat[4:8, 0:4] = - mr.SE3.hat(vel[0])
  
  np.testing.assert_array_equal(res.tan_map(), mat)
  
def test_cmtm_se3_vec2d_tan_map():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(2,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(12)
  mat[8:12, 8:12] *= 0.5
  mat[4:8, 0:4] = mat[8:12, 4:8] = - mr.SE3.hat(vel[0])
  mat[8:12, 4:8] *= 0.5
  mat[8:12, 0:4] = (- (mr.SE3.hat(vel[1]) - mr.SE3.hat(vel[0]) @ mr.SE3.hat(vel[0]))) * 0.5

  np.testing.assert_array_equal(res.tan_map(), mat)
  
def test_cmtm_se3_tan_map_adj():
  se3 = mr.SE3.rand() 
  
  res = mr.CMTM[mr.SE3](se3)
  
  mat = np.eye(6)
  
  np.testing.assert_array_equal(res.tan_map_adj(), mat)
  
def test_cmtm_se3_vec1d_tan_map_adj():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(1,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(12)
  mat[6:12, 0:6] = - mr.SE3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.tan_map_adj(), mat)
  
def test_cmtm_se3_vec2d_tan_map_adj():
  se3 = mr.SE3.rand()  
  vel = np.random.rand(2,6)
  
  res = mr.CMTM[mr.SE3](se3, vel)
  
  mat = np.eye(18)
  mat[12:18, 12:18] *= 0.5
  mat[6:12, 0:6] = mat[12:18, 6:12] = - mr.SE3.hat_adj(vel[0])
  mat[12:18, 6:12] *= 0.5
  mat[12:18, 0:6] = (- (mr.SE3.hat_adj(vel[1]) - mr.SE3.hat_adj(vel[0]) @ mr.SE3.hat_adj(vel[0]))) * 0.5

  np.testing.assert_array_equal(res.tan_map_adj(), mat)
  
def test_cmtm_se3_tan_inv():
  se3 = mr.SE3.rand()  
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(4*(i+1))

    np.testing.assert_allclose(res.tan_map() @ res.tan_map_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_tan_inv_adj():
  se3 = mr.SE3.rand() 
  
  for i in range(3):
    vel = np.random.rand(i,6)

    res = mr.CMTM[mr.SE3](se3, vel)
    
    mat = np.eye(6*(i+1))

    np.testing.assert_allclose(res.tan_map_adj() @ res.tan_map_inv_adj(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_se3_sub_vec():
  mat1 = mr.CMTM.rand(mr.SE3)
  mat2 = mr.CMTM.rand(mr.SE3)

  res = mr.CMTM.sub_vec(mat1, mat2, "bframe")
  sol = np.zeros(18)
  sol[0:6] = mr.SE3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
  sol[6:12] = mat2._vecs[0] - mat1._vecs[0]
  sol[12:18] = mat2._vecs[1] - mat1._vecs[1]

  np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)
    
def test_cmtm_se3_sub_ptan_vec():
  mat1 = mr.CMTM.rand(mr.SE3)
  mat2 = mr.CMTM.rand(mr.SE3)

  res = mr.CMTM.sub_ptan_vec(mat1, mat2, "bframe")
  sol = np.zeros(18)
  sol[ 0: 6] = mr.SE3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
  sol[ 6:12] = (mat2._vecs[0] - mat1._vecs[0]) + mr.SE3.hat_adj(sol[0:6]) @ mat1._vecs[0]
  sol[12:18] = (mat2._vecs[1] - mat1._vecs[1] + mr.SE3.hat_adj(sol[0:6]) @ mat1._vecs[1] + mr.SE3.hat_adj(sol[6:12]) @ mat1._vecs[0])

  np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15) 

def test_cmtm_se3_matmul():
  m1 = mr.CMTM.rand(mr.SE3, 3)
  m2 = mr.CMTM.rand(mr.SE3, 3)

  result = m1 @ m2

  expected_frame = m1.elem_mat() @ m2.elem_mat()
  expected_veloc = m2._mat.mat_inv_adj() @ m1.elem_vecs(0) + m2.elem_vecs(0)
  expected_accel = \
    m2._mat.mat_inv_adj() @ m1.elem_vecs(1) +\
    mr.SE3.hat_adj( m2._mat.mat_inv_adj() @ m1.elem_vecs(0) ) @ m2.elem_vecs(0) +\
    m2.elem_vecs(1)
    
  assert np.allclose(result.elem_mat(), expected_frame)
  assert np.allclose(result.elem_vecs(0), expected_veloc)
  assert np.allclose(result.elem_vecs(1), expected_accel)

def test_cmtm_se3_multiply():
  m1 = mr.CMTM.rand(mr.SE3, 1)
  m2 = mr.CMTM.rand(mr.SE3, 1)

  expected_frame = m1 @ m2
  result_mat = m1 @ m2.mat()

  np.testing.assert_allclose(expected_frame.mat(), result_mat, rtol=1e-15, atol=1e-15)