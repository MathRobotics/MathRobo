import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_so3():
  res = mr.SO3()
  
  np.testing.assert_array_equal(res.mat(), np.eye(3))
  
def test_so3_inv():
  rot = mr.SO3.rand()
  res = rot @ rot.inv()
  
  np.testing.assert_allclose(res.mat(), np.identity(3), rtol=1e-15, atol=1e-15)
  
def test_so3_adj():
  res = mr.SO3.rand()
  
  np.testing.assert_array_equal(res.mat_adj(), res.mat())
  
def test_so3_inv_adj():
  rot = mr.SO3.rand()
  res = rot.mat_adj() @ rot.mat_inv_adj()
  
  e = np.identity(3)
  
  np.testing.assert_allclose(res, e, rtol=1e-15, atol=1e-15)

def test_so3_hat():
  v = np.random.rand(3)  
  m = np.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])
  
  res = mr.SO3.hat(v)

  np.testing.assert_array_equal(res, m)
  
def test_so3_hat_commute():
  v1 = np.random.rand(3)
  v2 = np.random.rand(3)
  
  res1 = mr.SO3.hat(v1) @ v2
  res2 = mr.SO3.hat_commute(v2) @ v1
  
  np.testing.assert_allclose(res1, res2, rtol=1e-15, atol=1e-15)
  
def test_so3_vee():
  v = np.random.rand(3)
  
  hat = mr.SO3.hat(v)
  res = mr.SO3.vee(hat)
  
  np.testing.assert_array_equal(v, res)

def test_so3_exp():
  v = np.random.rand(3)
  a = np.random.rand()
  res = mr.SO3.exp(v, a)

  m = expm(a*mr.SO3.hat(v))
  
  np.testing.assert_allclose(res, m)
  
def test_so3_exp_integ():
  v = np.random.rand(3)
  a = np.random.rand()
  res = mr.SO3.exp_integ(v, a)

  def integrad(s):
    return expm(s*mr.SO3.hat(v))
  
  m, _ = integrate.quad_vec(integrad, 0, a)
  
  np.testing.assert_allclose(res, m)
  
def test_so3_exp_integ2nd():
  v = np.random.rand(3)
  a = np.random.rand()
  res = mr.SO3.exp_integ2nd(v, a)

  def integrad(s_):
    def integrad_(s):
      return expm(s*mr.SO3.hat(v))
    
    m, _ = integrate.quad_vec(integrad_, 0, s_)
    return m
  
  mat, _ = integrate.quad_vec(integrad, 0, a)
  
  np.testing.assert_allclose(res, mat)

def test_so3_sub_tan_vec():
  
  rot1 = mr.SO3.rand()
  rot2 = mr.SO3.rand()
  
  res = mr.SO3.sub_tan_vec(rot1, rot2, "bframe")
  sol = mr.SO3.vee(rot1.mat_inv() @ (rot2.mat() - rot1.mat()))
  
  np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

  res = mr.SO3.sub_tan_vec(rot1, rot2, "fframe")
  sol = mr.SO3.vee((rot2.mat() - rot1.mat()) @ rot1.mat_inv())
  
  np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)
  
def test_so3_matmul():  
  r = mr.SO3.rand().mat()
  
  rot1 = mr.SO3(r)
  rot2 = mr.SO3(r.transpose())
  res = rot1@rot2
  
  np.testing.assert_allclose(res.mat(), np.eye(3), rtol=1e-15, atol=1e-15)

def test_so3_matmul_mat():
  r = mr.SO3.rand().mat()
  
  rot1 = mr.SO3(r)
  rot2 = mr.SO3(r.transpose())
  res = rot1@rot2.mat()
  
  np.testing.assert_allclose(res, np.eye(3), rtol=1e-15, atol=1e-15)

def test_so3_matmul_vec():
  r = mr.SO3.rand().mat()
  vec = np.random.rand(3)
  
  rot = mr.SO3(r)
  res = rot@vec

  np.testing.assert_allclose(res, r @ vec, rtol=1e-15, atol=1e-15)
  
def test_so3_jac_lie_wrt_scaler():
  v = np.random.rand(3)
  dv = np.random.rand(3)
  a = np.random.rand()
  eps = 1e-8
  
  res = mr.jac_lie_wrt_scaler(mr.SO3, v, a, dv)
  
  r = mr.SO3.exp(v, a)
  v_ = v + dv*eps
  r_ = mr.SO3.exp(v_, a)
  
  dr = (r_ - r) / eps
  
  np.testing.assert_allclose(res, dr, 1e-3)
  
def test_so3_jac_lie_wrt_scaler_integ():
  v = np.random.rand(3)
  dv = np.random.rand(3)
  a = np.random.rand()
  eps = 1e-8
  
  def integrad(s):
    return mr.jac_lie_wrt_scaler(mr.SO3, v, s, dv)
  
  res, _ = integrate.quad_vec(integrad, 0, a)
  
  r = mr.SO3.exp_integ(v, a)
  v_ = v + dv*eps
  r_ = mr.SO3.exp_integ(v_, a)
  
  dr = (r_ - r) / eps
  
  np.testing.assert_allclose(res, dr, 1e-3)