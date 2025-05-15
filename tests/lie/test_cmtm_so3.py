import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_cmtm_so3():
  so3 = mr.SO3.rand()
  res = mr.CMTM[mr.SO3](so3)

  np.testing.assert_array_equal(res.mat(), so3.mat())
  
def test_cmtm_so3_vec1d():
  so3 = mr.SO3.rand()  
  vel = np.random.rand(1,3) 

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = so3.mat()
  mat[3:6,0:3] = so3.mat() @ so3.hat(vel[0])

  np.testing.assert_array_equal(res.mat(), mat)
  
def test_cmtm_so3_vec2d():
  so3 = mr.SO3.rand()  
  vec = np.random.rand(2,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  
  mat = np.zeros((9,9))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = so3.mat()
  mat[3:6,0:3] = mat[6:9,3:6] = so3.mat() @ so3.hat(vec[0])
  mat[6:9,0:3] = so3.mat() @ (so3.hat(vec[1]) + so3.hat(vec[0]) @ so3.hat(vec[0])) * 0.5

  np.testing.assert_allclose(res.mat(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_vec3d():
  so3 = mr.SO3.rand()  
  vec = np.random.rand(3,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  
  mat = np.zeros((12,12))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = mat[9:12,9:12] = so3.mat()
  mat[3:6,0:3] = mat[6:9,3:6] = mat[9:12,6:9] = so3.mat() @ so3.hat(vec[0])
  mat[6:9,0:3] = mat[9:12,3:6] =  so3.mat() @ (so3.hat(vec[1]) + so3.hat(vec[0]) @ so3.hat(vec[0])) * 0.5
  tmp1 = so3.mat() @ so3.hat(vec[2])
  tmp2 = so3.mat() @ so3.hat(vec[0]) @ so3.hat(vec[1])
  tmp3 = so3.mat() @ (so3.hat(vec[1]) + so3.hat(vec[0]) @ so3.hat(vec[0])) * 0.5 @ so3.hat(vec[0])
  mat[9:12,0:3] = ( tmp1 + tmp2 + tmp3 ) / 3.0

  np.testing.assert_allclose(res.mat(), mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_so3_adj():
  so3 = mr.SO3.rand()
  res = mr.CMTM[mr.SO3](so3)

  np.testing.assert_array_equal(res.mat_adj(), so3.mat_adj())
  
def test_cmtm_so3_vec1d():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(1,3) 

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = so3.mat_adj()
  mat[3:6,0:3] = so3.mat_adj() @ so3.hat_adj(vel[0])

  np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_so3_adj_vec2d():
  so3 = mr.SO3.rand()
  vec = np.random.rand(2,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  
  mat = np.zeros((9,9))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = so3.mat_adj()
  mat[3:6,0:3] = mat[6:9,3:6] = so3.mat_adj() @ so3.hat_adj(vec[0])
  mat[6:9,0:3] = so3.mat_adj() @ (so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) * 0.5

  np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_adj_vec3d():
  so3 = mr.SO3.rand()
  vec = np.random.rand(3,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  
  mat = np.zeros((12,12))
  mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = mat[9:12,9:12] = so3.mat_adj()
  mat[3:6,0:3] = mat[6:9,3:6] = mat[9:12,6:9] = so3.mat_adj() @ so3.hat_adj(vec[0])
  mat[6:9,0:3] = mat[9:12,3:6] = so3.mat_adj() @ (so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) * 0.5
  tmp1 = so3.mat_adj() @ so3.hat_adj(vec[2])
  tmp2 = so3.mat_adj() @ so3.hat_adj(vec[0]) @ so3.hat_adj(vec[1])
  tmp3 = so3.mat_adj() @ (so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) * 0.5 @ so3.hat_adj(vec[0])
  mat[9:12,0:3] = ( tmp1 + tmp2 + tmp3 ) / 3.0

  np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_so3_getter():
  n = 5

  so3 = mr.SO3.rand() 
  vec = np.random.rand(n,3)

  res = mr.CMTM[mr.SO3](so3,vec)
  
  np.testing.assert_array_equal(res.elem_mat(), so3.mat())
  for i in range(n):
    np.testing.assert_array_equal(res.elem_vecs(i), vec[i])

def test_cmtm_so3_set_mat():
  n = 5

  so3 = mr.SO3.rand() 
  vec = np.random.rand(n,3)

  cmtm = mr.CMTM[mr.SO3](so3,vec)

  res = mr.CMTM.set_mat(mr.SO3, cmtm.mat())

  np.testing.assert_allclose(res.elem_mat(), cmtm.elem_mat(), rtol=1e-15, atol=1e-15)
  for i in range(n):
    np.testing.assert_allclose(res.elem_vecs(i), cmtm.elem_vecs(i), rtol=1e-15, atol=1e-15)

def test_cmtm_so3_inv():
  so3 = mr.SO3.rand()
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)

    expected_mat = np.eye(3*(i+1))
    result_mat = res @ res.inv()

    np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_so3_mat_inv():
  so3 = mr.SO3.rand()
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    expected_mat = np.eye(3*(i+1))
    result_mat = res.mat() @ res.mat_inv()

    np.testing.assert_allclose(result_mat, expected_mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_so3_inv_adj():
  so3 = mr.SO3.rand() 
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.mat_adj() @ res.mat_inv_adj(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_hat():
  vec = np.random.rand(1,3)

  res = mr.CMTM.hat(mr.SO3, vec)
  mat = mr.SO3.hat(vec[0])

  np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_hat2():
  vec = np.random.rand(2,3)

  res = mr.CMTM.hat(mr.SO3, vec)
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = mr.SO3.hat(vec[0])
  mat[3:6,0:3] = mr.SO3.hat(vec[1])  

  np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_hat_adj():
  vec = np.random.rand(1,3)

  res = mr.CMTM.hat_adj(mr.SO3, vec)
  mat = mr.SO3.hat_adj(vec[0])

  np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_hat_adj_2():
  vec = np.random.rand(2,3)

  res = mr.CMTM.hat_adj(mr.SO3, vec)
  mat = np.zeros((6,6))
  mat[0:3,0:3] = mat[3:6,3:6] = mr.SO3.hat_adj(vec[0])
  mat[3:6,0:3] = mr.SO3.hat_adj(vec[1])  

  np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_ptan_map():
  so3 = mr.SO3.rand()
  
  res = mr.CMTM[mr.SO3](so3)
  
  np.testing.assert_array_equal(res.ptan_map(), np.eye(3))

def test_cmtm_so3_vec1d_ptan_map():
  so3 = mr.SO3.rand()  
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat(vel[0])
  
  np.testing.assert_array_equal(res.ptan_map(), mat)

def test_cmtm_so3_vec2d_ptan_map():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat(vel[0])
  mat[6:9, 0:3] = - (mr.SO3.hat(vel[1]) - mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0])) 

  np.testing.assert_array_equal(res.ptan_map(), mat)

def test_cmtm_so3_vec3d_ptan_map():
  so3 = mr.SO3.rand()
  vel = np.random.rand(3,3)

  res = mr.CMTM[mr.SO3](so3, vel)
   
  mat = np.eye(12)
  mat[3:6, 0:3] = mat[6:9, 3:6] = mat[9:12, 6:9] = - mr.SO3.hat(vel[0])
  mat[6:9, 0:3] = mat[9:12, 3:6] = - (mr.SO3.hat(vel[1]) - mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0])) 
  mat[9:12, 0:3] = - (mr.SO3.hat(vel[2]) - mr.SO3.hat(vel[1]) @ mr.SO3.hat(vel[0]) - mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[1]) + mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0]))

  np.testing.assert_allclose(res.ptan_map(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_ptan_map_adj():
  so3 = mr.SO3.rand()  
  
  res = mr.CMTM[mr.SO3](so3)
  
  np.testing.assert_array_equal(res.ptan_map_adj(), np.eye(3))

def test_cmtm_so3_vec1d_ptan_map_adj():
  so3 = mr.SO3.rand()  
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.ptan_map_adj(), mat)

def test_cmtm_so3_vec2d_ptan_map_adj():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 0:3] = - (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.ptan_map_adj(), mat)

def test_cmtm_so3_vec3d_ptan_map_adj():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(3,3)

  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(12)
  mat[3:6, 0:3] = mat[6:9, 3:6] = mat[9:12, 6:9] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 0:3] = mat[9:12, 3:6] = - (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 
  mat[9:12, 0:3] = - (mr.SO3.hat_adj(vel[2]) - mr.SO3.hat_adj(vel[1]) @ mr.SO3.hat_adj(vel[0]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[1]) + mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0]))

  np.testing.assert_allclose(res.ptan_map_adj(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_ptan_to_tan():
  n = 5

  for i in range(n):
    so3 = mr.SO3.rand()
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3)

    for j in range(i):
      np.testing.assert_allclose(res.ptan_to_tan(3, i)[3*j:3*(j+1),3*j:3*(j+1)], mat, rtol=1e-15, atol=1e-15)
      mat = mat / (j+1)

def test_cmtm_so3_tan_to_ptan():
  n = 5

  for i in range(n):
    so3 = mr.SO3.rand()
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3)

    for j in range(i):
      np.testing.assert_allclose(res.tan_to_ptan(3, i)[3*j:3*(j+1),3*j:3*(j+1)], mat, rtol=1e-15, atol=1e-15)
      mat = mat * (j+1)

def test_cmtm_so3_tan_map():
  so3 = mr.SO3.rand()
  
  res = mr.CMTM[mr.SO3](so3)
  
  np.testing.assert_array_equal(res.tan_map(), np.eye(3))
  
def test_cmtm_so3_vec1d_tan_map():
  so3 = mr.SO3.rand()  
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat(vel[0])
  
  np.testing.assert_array_equal(res.tan_map(), mat)
  
def test_cmtm_so3_vec2d_tan_map():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[6:9, 6:9] *= 0.5
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat(vel[0])
  mat[6:9, 3:6] *= 0.5
  mat[6:9, 0:3] = - 0.5 * (mr.SO3.hat(vel[1]) - mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0])) 

  np.testing.assert_array_equal(res.tan_map(), mat)

def test_cmtm_so3_vec3d_tan_map():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[6:9, 6:9] *= 0.5
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat(vel[0])
  mat[6:9, 3:6] *= 0.5
  mat[6:9, 0:3] = - 0.5 * (mr.SO3.hat(vel[1]) - mr.SO3.hat(vel[0]) @ mr.SO3.hat(vel[0])) 

  np.testing.assert_array_equal(res.tan_map(), mat)
  
def test_cmtm_so3_tan_map_adj():
  so3 = mr.SO3.rand()  
  
  res = mr.CMTM[mr.SO3](so3)
  
  np.testing.assert_array_equal(res.tan_map_adj(), np.eye(3))
  
def test_cmtm_so3_vec1d_tan_map_adj():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.tan_map_adj(), mat)
  
def test_cmtm_so3_vec2d_tan_map_adj():
  so3 = mr.SO3.rand()
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[6:9, 6:9] *= 0.5
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 3:6] *= 0.5
  mat[6:9, 0:3] = - 0.5 * (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.tan_map_adj(), mat)
  
def test_cmtm_so3_tan_inv():
  so3 = mr.SO3.rand()  
  
  for i in range(3):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.tan_map() @ res.tan_map_inv(), mat, rtol=1e-15, atol=1e-15)
    
def test_cmtm_so3_tan_inv_adj():
  so3 = mr.SO3.rand()  
  
  for i in range(3):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.tan_map_adj() @ res.tan_map_inv_adj(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_sub_vec():
  mat1 = mr.CMTM.rand(mr.SO3)
  mat2 = mr.CMTM.rand(mr.SO3)

  res = mr.CMTM.sub_vec(mat1, mat2, "bframe")
  sol = np.zeros(9)
  sol[0:3] = mr.SO3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
  sol[3:6] = mat2._vecs[0] - mat1._vecs[0]
  sol[6:9] = mat2._vecs[1] - mat1._vecs[1]

  np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_sub_ptan_vec():
  mat1 = mr.CMTM.rand(mr.SO3)
  mat2 = mr.CMTM.rand(mr.SO3)

  res = mr.CMTM.sub_ptan_vec(mat1, mat2, "bframe")
  sol = np.zeros(9)
  sol[0:3] = mr.SO3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
  sol[3:6] = (mat2._vecs[0] - mat1._vecs[0]) + mr.SO3.hat_adj(sol[0:3]) @ mat1._vecs[0]
  sol[6:9] = (mat2._vecs[1] - mat1._vecs[1] + mr.SO3.hat_adj(sol[0:3]) @ mat1._vecs[1] + mr.SO3.hat_adj(sol[3:6]) @ mat1._vecs[0])

  np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15) 

def test_cmtm_so3_matmul():
  m1 = mr.CMTM.rand(mr.SO3, 3)
  m2 = mr.CMTM.rand(mr.SO3, 3)

  result = m1 @ m2

  expected_frame = m1.elem_mat() @ m2.elem_mat()
  expected_veloc = m2._mat.mat_inv_adj() @ m1.elem_vecs(0) + m2.elem_vecs(0)
  expected_accel = \
    m2._mat.mat_inv_adj() @ m1.elem_vecs(1) +\
    mr.SO3.hat_adj( m2._mat.mat_inv_adj() @ m1.elem_vecs(0) ) @ m2.elem_vecs(0) +\
    m2.elem_vecs(1)
    
  assert np.allclose(result.elem_mat(), expected_frame)
  assert np.allclose(result.elem_vecs(0), expected_veloc)
  assert np.allclose(result.elem_vecs(1), expected_accel)

def test_cmtm_so3_multiply():
  m1 = mr.CMTM.rand(mr.SO3, 5)
  m2 = mr.CMTM.rand(mr.SO3, 5)

  result_mat = m1 @ m2

  expected_mat = m1.mat() @ m2.mat()

  np.testing.assert_allclose(expected_mat, result_mat.mat(), rtol=1e-15, atol=1e-15)

def test_cmtm_so3_multiply_adj():
  m1 = mr.CMTM.rand(mr.SO3, 5)
  m2 = mr.CMTM.rand(mr.SO3, 5)

  result_mat = m1 @ m2

  expected_mat = m1.mat_adj() @ m2.mat_adj()

  np.testing.assert_allclose(expected_mat, result_mat.mat_adj(), rtol=1e-15, atol=1e-15)