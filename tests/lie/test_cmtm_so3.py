import numpy as np

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
  tmp1 = so3.mat() @ so3.hat(vec[2]/2)
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
  tmp1 = so3.mat_adj() @ so3.hat_adj(vec[2]/2)
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

  np.testing.assert_allclose(res.elem_mat(), cmtm.elem_mat(), rtol=1e-14, atol=1e-14)
  for i in range(n):
    np.testing.assert_allclose(res.elem_vecs(i), cmtm.elem_vecs(i), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(res.elem_vecs(i), vec[i], rtol=1e-14, atol=1e-14)

def test_cmtm_so3_vecs():
  n = 5
  so3 = mr.SO3.rand()
  vec = np.random.rand(n,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  np.testing.assert_array_equal(res.vecs(), vec)
  np.testing.assert_array_equal(res.vecs(3), vec[:2])

def test_cmtm_so3_vecs_flatten():
  n = 5
  so3 = mr.SO3.rand()
  vec = np.random.rand(n,3)

  res = mr.CMTM[mr.SO3](so3, vec)
  np.testing.assert_array_equal(res.vecs_flatten(), vec.flatten())
  np.testing.assert_array_equal(res.vecs_flatten(3), vec[:2].flatten())

def test_cmtm_so3_tan_vecs():
  n = 3

  so3 = mr.SO3.rand()
  vec = np.random.rand(n,3)

  mat = mr.CMTM[mr.SO3](so3, vec)

  res = mat.tan_vecs()

  np.testing.assert_allclose(res[0], vec[0])
  np.testing.assert_allclose(res[1], vec[1] + mr.SO3.hat_adj(vec[0]) @ vec[0])
  np.testing.assert_allclose(res[2], 0.5 * (vec[2] + mr.SO3.hat_adj(vec[1]) @ vec[0] + mr.SO3.hat_adj(vec[0]) @ vec[1] + mr.SO3.hat_adj(vec[0]) @ mr.SO3.hat_adj(vec[0]) @ vec[0]) )

def test_cmtm_so3_tan_vecs_flatten():
  n = 3

  so3 = mr.SO3.rand()
  vec = np.random.rand(n,3)

  mat = mr.CMTM[mr.SO3](so3, vec)

  res = mat.tan_vecs_flatten()

  np.testing.assert_allclose(res[ :3], vec[0])
  np.testing.assert_allclose(res[3:6], vec[1] + mr.SO3.hat_adj(vec[0]) @ vec[0])
  np.testing.assert_allclose(res[6:9], 0.5 * (vec[2] + mr.SO3.hat_adj(vec[1]) @ vec[0] + mr.SO3.hat_adj(vec[0]) @ vec[1] + mr.SO3.hat_adj(vec[0]) @ mr.SO3.hat_adj(vec[0]) @ vec[0]) )

def test_cmtm_so3_ptan_vecs():
  n = 3

  so3 = mr.SO3.rand()
  vec = np.random.rand(n,3)

  mat = mr.CMTM[mr.SO3](so3, vec)

  res = mat.ptan_vecs()

  np.testing.assert_allclose(res[0], vec[0])
  np.testing.assert_allclose(res[1], vec[1] + mr.SO3.hat_adj(vec[0]) @ vec[0])
  np.testing.assert_allclose(res[2], vec[2] + mr.SO3.hat_adj(vec[1]) @ vec[0] + mr.SO3.hat_adj(vec[0]) @ vec[1] + mr.SO3.hat_adj(vec[0]) @ mr.SO3.hat_adj(vec[0]) @ vec[0])

def test_cmtm_so3_tan_vecs_flatten():
  n = 3

  so3 = mr.SO3.rand()
  vec = np.random.rand(n,3)

  mat = mr.CMTM[mr.SO3](so3, vec)

  res = mat.ptan_vecs_flatten()

  np.testing.assert_allclose(res[ :3], vec[0])
  np.testing.assert_allclose(res[3:6], vec[1] + mr.SO3.hat_adj(vec[0]) @ vec[0])
  np.testing.assert_allclose(res[6:9], vec[2] + mr.SO3.hat_adj(vec[1]) @ vec[0] + mr.SO3.hat_adj(vec[0]) @ vec[1] + mr.SO3.hat_adj(vec[0]) @ mr.SO3.hat_adj(vec[0]) @ vec[0])

def test_cmtm_so3_inv():
  so3 = mr.SO3.rand()
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)

    expected_mat = np.eye(3*(i+1))
    result_mat = res @ res.inv()

    np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-15, atol=1e-15)

    result_mat = res.inv() @ res
    np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-15, atol=1e-15)
  
def test_cmtm_so3_mat_inv():
  so3 = mr.SO3.rand()
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    expected_mat = np.eye(3*(i+1))
    result_mat = res.mat() @ res.mat_inv()

    np.testing.assert_allclose(result_mat, expected_mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_mat_inv_elem():
  so3 = mr.SO3.rand()
  res = mr.CMTM[mr.SO3](so3)

  result = res.mat_inv()
  expected = so3.mat_inv()

  np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

  vec = np.random.rand(1,3)
  res = mr.CMTM[mr.SO3](so3, vec)

  result = res.mat_inv()
  expected = np.zeros((6,6))
  expected[0:3,0:3] = expected[3:6,3:6] = so3.mat_inv()
  expected[3:6,0:3] = -so3.hat(vec[0]) @ so3.mat_inv()

  np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

  vec = np.random.rand(2,3)
  res = mr.CMTM[mr.SO3](so3, vec)

  result = res.mat_inv()
  expected = np.zeros((9,9))
  expected[0:3,0:3] = expected[3:6,3:6] = expected[6:9,6:9] = so3.mat_inv()
  expected[3:6,0:3] = expected[6:9,3:6] = -so3.hat(vec[0]) @ so3.mat_inv()
  expected[6:9,0:3] = (-so3.hat(vec[1]) + so3.hat(vec[0]) @ so3.hat(vec[0])) @ so3.mat_inv() * 0.5

  np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)
    
def test_cmtm_so3_inv_adj():
  so3 = mr.SO3.rand() 
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.mat_adj() @ res.mat_inv_adj(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_mat_inv_adj_elem():
  so3 = mr.SO3.rand()
  res = mr.CMTM[mr.SO3](so3)

  result = res.mat_inv_adj()
  expected = so3.mat_inv_adj()

  np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

  vec = np.random.rand(1,3)
  res = mr.CMTM[mr.SO3](so3, vec)

  result = res.mat_inv_adj()
  expected = np.zeros((6,6))
  expected[0:3,0:3] = expected[3:6,3:6] = so3.mat_inv_adj()
  expected[3:6,0:3] = -so3.hat_adj(vec[0]) @ so3.mat_inv_adj()

  np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

  vec = np.random.rand(2,3)
  res = mr.CMTM[mr.SO3](so3, vec)

  result = res.mat_inv_adj()
  expected = np.zeros((9,9))
  expected[0:3,0:3] = expected[3:6,3:6] = expected[6:9,6:9] = so3.mat_inv_adj()
  expected[3:6,0:3] = expected[6:9,3:6] = -so3.hat_adj(vec[0]) @ so3.mat_inv_adj()
  expected[6:9,0:3] = (-so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) @ so3.mat_inv_adj() * 0.5

  np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_hat():
  n = 5
  for i in range(1,n+1):
    vec = np.random.rand(i,3)

    res = mr.CMTM.hat(mr.SO3, vec)
    mat = np.zeros((3*i,3*i))
    for j in range(i):
      for k in range(j, i):
        mat[3*k:3*(k+1), 3*(k-j):3*(k-j+1)] = mr.SO3.hat(vec[j])

    np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_hat_adj():
  n = 5
  for i in range(1,n+1):
    vec = np.random.rand(i,3)

    res = mr.CMTM.hat_adj(mr.SO3, vec)
    mat = np.zeros((3*i,3*i))
    for j in range(i):
      for k in range(j, i):
        mat[3*k:3*(k+1), 3*(k-j):3*(k-j+1)] = mr.SO3.hat_adj(vec[j])

    np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_vee():
  n = 5
  for i in range(1,n+1):
    vec = np.random.rand(i,3)

    mat = mr.CMTM.hat(mr.SO3, vec)
    res = mr.CMTM.vee(mr.SO3, mat)

    np.testing.assert_allclose(res, vec) 

def test_cmtm_so3_vee_adj():
  n = 5
  for i in range(1,n+1):
    vec = np.random.rand(i,3)

    mat = mr.CMTM.hat_adj(mr.SO3, vec)
    res = mr.CMTM.vee_adj(mr.SO3, mat)

    np.testing.assert_allclose(res, vec) 

def test_cmtm_so3_ptan_map():
  so3 = mr.SO3.rand()
  
  res = mr.CMTM[mr.SO3](so3)
  
  np.testing.assert_array_equal(res.ptan_map(), np.eye(3))

def test_cmtm_so3_vec1d_ptan_map():
  so3 = mr.SO3.rand()  
  vel = np.random.rand(1,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(6)
  mat[3:6, 0:3] = - mr.SO3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.ptan_map(), mat)

def test_cmtm_so3_vec2d_ptan_map():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 0:3] = - (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.ptan_map(), mat)

def test_cmtm_so3_vec3d_ptan_map():
  so3 = mr.SO3.rand()
  vel = np.random.rand(3,3)

  res = mr.CMTM[mr.SO3](so3, vel)
   
  mat = np.eye(12)
  mat[3:6, 0:3] = mat[6:9, 3:6] = mat[9:12, 6:9] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 0:3] = mat[9:12, 3:6] = - (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 
  mat[9:12, 0:3] = - (mr.SO3.hat_adj(vel[2]) - mr.SO3.hat_adj(vel[1]) @ mr.SO3.hat_adj(vel[0]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[1]) + mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0]))

  np.testing.assert_allclose(res.ptan_map(), mat, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_ptan_map_ptan_vec():
  n = 5
  for i in range(2,n):
    res = mr.CMTM.rand(mr.SO3,i)

    vec = res.ptan_map(i-1) @ res.vecs_flatten()

    np.testing.assert_allclose(vec, res.ptan_vecs_flatten(), rtol=1e-15, atol=1e-15)

def test_cmtm_so3_ptan_map_inv_ptan_vec():
  n = 5
  for i in range(2,n):
    res = mr.CMTM.rand(mr.SO3,i)

    vec = res.ptan_map_inv(i-1) @ res.ptan_vecs_flatten()

    np.testing.assert_allclose(vec, res.vecs_flatten(), rtol=1e-15, atol=1e-15)

def test_cmtm_so3_ptan_inv():
  n = 5
  for i in range(n):
    res = mr.CMTM.rand(mr.SO3,i+1)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.ptan_map() @ res.ptan_map_inv(), mat, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(res.ptan_map_inv() @ res.ptan_map(), mat, rtol=1e-14, atol=1e-14)

def test_cmtm_so3_ptan_to_tan():
  n = 5

  for i in range(n):
    res = mr.CMTM.rand(mr.SO3, i+1)
    
    mat = np.eye(3)

    for j in range(i):
      np.testing.assert_allclose(res.ptan_to_tan(3, i)[3*j:3*(j+1),3*j:3*(j+1)], mat, rtol=1e-15, atol=1e-15)
      mat = mat / (j+1)

def test_cmtm_so3_tan_to_ptan():
  n = 5

  for i in range(n):
    res = mr.CMTM.rand(mr.SO3, i+1)
    
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
  mat[3:6, 0:3] = - mr.SO3.hat_adj(vel[0])
  
  np.testing.assert_array_equal(res.tan_map(), mat)
  
def test_cmtm_so3_vec2d_tan_map():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(2,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(9)
  mat[6:9, 6:9] *= 0.5
  mat[3:6, 0:3] = mat[6:9, 3:6] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 3:6] *= 0.5
  mat[6:9, 0:3] = - 0.5 * (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 

  np.testing.assert_array_equal(res.tan_map(), mat)

def test_cmtm_so3_vec3d_tan_map():
  so3 = mr.SO3.rand() 
  vel = np.random.rand(3,3)
  
  res = mr.CMTM[mr.SO3](so3, vel)
  
  mat = np.eye(12)
  mat[6:9, 6:9] *= 0.5
  mat[9:12, 9:12] /= 6
  mat[3:6, 0:3] = mat[6:9, 3:6] = mat[9:12, 6:9] = - mr.SO3.hat_adj(vel[0])
  mat[6:9, 3:6] *= 0.5
  mat[9:12, 6:9] /= 6
  mat[6:9, 0:3] = - 0.5 * (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) 
  mat[9:12, 3:6] = - (mr.SO3.hat_adj(vel[1]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) / 6
  mat[9:12, 0:3] = - (mr.SO3.hat_adj(vel[2]) - mr.SO3.hat_adj(vel[1]) @ mr.SO3.hat_adj(vel[0]) - mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[1]) + mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0]) @ mr.SO3.hat_adj(vel[0])) / 6

  np.testing.assert_allclose(res.tan_map(), mat)

def test_cmtm_so3_tan_map_tan_vec():
  n = 6
  for i in range(2,n):
    res = mr.CMTM.rand(mr.SO3,i)

    vec = res.tan_map(i-1) @ res.vecs_flatten()

    np.testing.assert_allclose(vec, res.tan_vecs_flatten(), rtol=1e-15, atol=1e-15)

def test_cmtm_so3_tan_map_inv_ptan_vec():
  n = 6
  for i in range(2,n):
    res = mr.CMTM.rand(mr.SO3,i)

    vec = res.tan_map_inv(i-1) @ res.tan_vecs_flatten()

    np.testing.assert_allclose(vec, res.vecs_flatten(), rtol=1e-15, atol=1e-15)

def test_cmtm_so3_tan_inv():
  so3 = mr.SO3.rand()  
  
  for i in range(5):
    vel = np.random.rand(i,3)

    res = mr.CMTM[mr.SO3](so3, vel)
    
    mat = np.eye(3*(i+1))

    np.testing.assert_allclose(res.tan_map() @ res.tan_map_inv(), mat, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(res.tan_map_inv() @ res.tan_map(), mat, rtol=1e-14, atol=1e-14)

def test_cmtm_so3_sub_vec():
  order = 5

  for i in range(order):
    mat1 = mr.CMTM.rand(mr.SO3, i+1)
    mat2 = mr.CMTM.rand(mr.SO3, i+1)

    res = mr.CMTM.sub_vec(mat1, mat2, "bframe")
    sol = np.zeros(3*(i+1))
    sol[0:3] = mr.SO3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
    if i > 0: 
      for j in range(i):
        sol[3*(j+1):3*(j+2)] = mat2._vecs[j] - mat1._vecs[j]
    
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

def test_cmtm_so3_sub():
  n = 6
  for i in range(n):
    mat1 = mr.CMTM.rand(mr.SO3, i+1)
    mat2 = mr.CMTM.rand(mr.SO3, i+1)

    vec1 = mr.CMTM.ptan_to_tan(mr.SO3.dof(), i+1) @ mr.CMTM.sub_ptan_vec(mat1, mat2, "bframe")
    vec2 = mat1.tan_map() @ mr.CMTM.sub_vec(mat1, mat2, "bframe")

    np.testing.assert_allclose(vec1, vec2, rtol=1e-15, atol=1e-15)


def test_cmtm_so3_sub_tan_vec():
  n = 5
  for i in range(n):
    mat1 = mr.CMTM.rand(mr.SO3, i+1)
    mat2 = mr.CMTM.rand(mr.SO3, i+1)

    res = mr.CMTM.sub_tan_vec(mat1, mat2, "bframe")
    vec = mr.CMTM.ptan_to_tan(mr.SO3.dof(), i+1) @ mr.CMTM.sub_ptan_vec(mat1, mat2, "bframe")

    np.testing.assert_allclose(res, vec, rtol=1e-15, atol=1e-15)

def test_cmtm_so3_matmul():
  order = 4
  m1 = mr.CMTM.rand(mr.SO3, order)
  m2 = mr.CMTM.rand(mr.SO3, order)

  result = m1 @ m2

  expected_frame = m1.elem_mat() @ m2.elem_mat()
  expected_veloc = m2._mat.mat_inv_adj() @ m1.elem_vecs(0) + m2.elem_vecs(0)
  expected_accel = \
    m2._mat.mat_inv_adj() @ m1.elem_vecs(1) +\
    mr.SO3.hat_adj( m2._mat.mat_inv_adj() @ m1.elem_vecs(0) ) @ m2.elem_vecs(0) +\
    m2.elem_vecs(1)

  expected_jerk = \
    m2._mat.mat_inv_adj() @ m1.elem_vecs(2) +\
    - 2* mr.SO3.hat_adj( m2.elem_vecs(0)) @ m2._mat.mat_inv_adj() @ m1.elem_vecs(1) \
    + (-mr.SO3.hat_adj(m2.elem_vecs(1)) + mr.SO3.hat_adj( m2.elem_vecs(0)) @ mr.SO3.hat_adj( m2.elem_vecs(0))) @ m2._mat.mat_inv_adj() @ m1.elem_vecs(0) \
    + m2.elem_vecs(2)

  assert np.allclose(result.elem_mat(), expected_frame)
  assert np.allclose(result.elem_vecs(0), expected_veloc)
  assert np.allclose(result.elem_vecs(1), expected_accel)
  assert np.allclose(result.elem_vecs(2), expected_jerk)

def test_cmtm_so3_multiply():
  for i in range(5):
    m1 = mr.CMTM.rand(mr.SO3, i+1)
    m2 = mr.CMTM.rand(mr.SO3, i+1)

    result_mat = m1 @ m2
    expected_mat = m1.mat() @ m2.mat()

    np.testing.assert_allclose(expected_mat, result_mat.mat(), rtol=1e-14, atol=1e-14)

def test_cmtm_so3_multiply_adj():
  for i in range(5):
    m1 = mr.CMTM.rand(mr.SO3, i+1)
    m2 = mr.CMTM.rand(mr.SO3, i+1)

    result_mat = m1 @ m2

    expected_mat = m1.mat_adj() @ m2.mat_adj()


    np.testing.assert_allclose(expected_mat, result_mat.mat_adj(), rtol=1e-14, atol=1e-14)

def test_cmtm_so3_multiply_and_vec():
  order = 5
  for i in range(order):
    x1 = mr.CMTM.rand(mr.SO3, i+1)
    x2 = mr.CMTM.rand(mr.SO3, i+1)

    result = x1 @ x2
    expected_tan_vec = x2.mat_inv_adj(i) @ x1.tan_vecs_flatten() + x2.tan_vecs_flatten()
    expected_vec = result.tan_map_inv(i) @  expected_tan_vec

    np.testing.assert_allclose(result.tan_vecs_flatten(), expected_tan_vec, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.vecs_flatten(), expected_vec, rtol=1e-10, atol=1e-10)