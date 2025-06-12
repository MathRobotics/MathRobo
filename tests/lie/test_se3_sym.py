import sympy as sp
import numpy as np

import mathrobo as mr

def test_se3_hat():
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  
  m = sp.Matrix([ \
    [0, -v[2], v[1], v[3]], \
    [v[2], 0, -v[0], v[4]], \
    [-v[1], v[0], 0, v[5]],
    [0,     0,    0,    0]])
  
  res = mr.SE3.hat(v, 'sympy')

  assert res == m
  
def test_se3_hat_commute():
  x = sp.symbols("x_{0:6}", Integer=True)
  y = sp.symbols("y_{0:6}", Integer=True)
  v = sp.Matrix(x)
  w = sp.Matrix(y)
  
  w_ = mr.zeros(4, 'sympy')
  w_[0:3,0] = w[0:3,0]
  
  res1 = mr.SE3.hat(v, 'sympy') @ w_
  res2 = mr.SE3.hat_commute(w, 'sympy') @ v
  
  assert res1 == res2
  
def test_se3_vee():
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)

  hat = mr.SE3.hat(v, 'sympy')
  res = mr.SE3.vee(hat, 'sympy')

  assert res == v

def test_se3_exp():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  
  r = mr.SE3.exp(v, a, 'sympy')

  angle = np.random.rand()
  vec = np.random.rand(6)
  
  vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )

  res = mr.sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle)]) 
  
  m = mr.SE3.exp(vec, angle)

  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))
  
def test_se3_exp_integ():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  
  r = mr.SE3.exp_integ(v, a, 'sympy')

  angle = np.random.rand()
  vec = np.random.rand(6)
  
  vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )

  res = mr.sympy_subs_mat(r, x, vec)
  res = res.subs([(a, angle)]) 
  
  m = mr.SE3.exp_integ(vec, angle)
  
  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))
  
def test_se3_jac_lie_wrt_scaler():
  a = sp.symbols('a')
  x = sp.symbols("x_{0:6}", Integer=True)
  v = sp.Matrix(x)
  dx = sp.symbols("dx_{0:6}", Integer=True)
  dv = sp.Matrix(dx)
  
  r = mr.jac_lie_wrt_scaler(mr.SE3, v, a, dv, 'sympy')

  angle = np.random.rand()
  vec = np.random.rand(6)
  dvec = np.random.rand(6)
  
  vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )
  
  res = mr.sympy_subs_mat(r, x, vec)
  res = mr.sympy_subs_mat(res, dx, dvec)
  res = res.subs([(a, angle)]) 
  
  m = mr.jac_lie_wrt_scaler(mr.SE3, vec, angle, dvec)
  
  np.testing.assert_allclose(m, mr.sympy_to_numpy(res))

# def test_se3_jac_lie_wrt_scaler_integ():
#   a_ = sp.symbols('a_')
#   a = sp.symbols('a')
#   x = sp.symbols("x_{0:6}", Integer=True)
#   v = sp.Matrix(x)
#   dx = sp.symbols("dx_{0:6}", Integer=True)
#   dv = sp.Matrix(dx)
  
#   r_ = mr.jac_lie_wrt_scaler(mr.SE3, v, a_, dv, 'sympy')
#   r = sp.integrate(r_, [a_, 0, a])
  
#   angle = np.random.rand()
#   vec = np.random.rand(6)
#   dvec = np.random.rand(6)
  
#   vec[0:3] = vec[0:3] / np.linalg.norm(vec[0:3] )
  
#   def integrad(s):
#     return mr.jac_lie_wrt_scaler(mr.SE3, vec, s, dvec)
  
#   m, _ = integrate.quad_vec(integrad, 0, angle)
  
#   res = mr.sympy_subs_mat(r, x, vec)
#   res = mr.sympy_subs_mat(res, dx, dvec)
#   res = res.subs([(a, angle)]) 
  
#   np.testing.assert_allclose(m, mr.sympy_to_numpy(res))