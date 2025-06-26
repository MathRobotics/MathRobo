import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_se3():
    res = mr.SE3()

    np.testing.assert_array_equal(res.mat(), np.identity(4))
    
def test_se3_inv():
    h = mr.SE3.rand()
    res = h @ h.inv()
    
    np.testing.assert_allclose(res.mat(), np.identity(4), rtol=1e-15, atol=1e-15)
    
def test_se3_adj():
    v = np.random.rand(6)
    v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
    r = mr.SO3.exp(v[0:3]) 
    
    res = mr.SE3(r, v[3:6])
    
    m = np.zeros((6,6))
    m[0:3, 0:3] = r 
    m[3:6,0:3] = mr.SO3.hat(v[3:6])@r
    m[3:6, 3:6] = r 
    
    np.testing.assert_allclose(res.mat_adj(), m)
    
def test_se3_set_adj():
    h = mr.SE3.rand()
    
    res = mr.SE3.set_mat_adj(h.mat_adj())
    
    np.testing.assert_allclose(res.mat(), h.mat())
    
def test_se3_inv_adj():
    h = mr.SE3.rand()
    
    res = h.mat_adj() @ h.mat_inv_adj()
    
    np.testing.assert_allclose(res, np.eye(6), rtol=1e-15, atol=1e-15)

def test_se3_hat():
    v = np.random.rand(6)  
    m = np.array([[0., -v[2], v[1], v[3]],
                                [v[2], 0., -v[0], v[4]],
                                [-v[1], v[0], 0., v[5]],
                                [  0.,  0.,  0.,  0.]])
    
    res = mr.SE3.hat(v)

    np.testing.assert_array_equal(res, m)
    
def test_se3_hat_commute():
    v1 = np.random.rand(6)
    v2 = np.append(np.random.rand(3), 0.)

    res1 = mr.SE3.hat(v1) @ v2
    res2 = mr.SE3.hat_commute(v2) @ v1
    
    np.testing.assert_allclose(res1, res2)
    
def test_se3_vee():
    v = np.random.rand(6)
    
    hat = mr.SE3.hat(v)
    res = mr.SE3.vee(hat)
    
    np.testing.assert_array_equal(v, res)

def test_se3_vee_adj():
    v = np.random.rand(6)
    
    hat = mr.SE3.hat_adj(v)
    res = mr.SE3.vee_adj(hat)
    
    np.testing.assert_array_equal(v, res)
    
def test_se3_exp():
    v = np.random.rand(6)
    a = np.random.rand()
    res = mr.SE3.exp(v, a)

    m = expm(a*mr.SE3.hat(v))
    
    np.testing.assert_allclose(res, m)
    
def test_se3_exp_integ():
    v = np.random.rand(6)
    a = np.random.rand()
    res = mr.SE3.exp_integ(v, a)

    def integrad(s):
        return expm(s*mr.SE3.hat(v))
    
    m, _ = integrate.quad_vec(integrad, 0, a)
    
    m[3,3] = m[3,3] / a
    
    np.testing.assert_allclose(res, m)
    
def test_se3_hat_adj():
    v = np.random.rand(6)  
    m = np.array([[0., -v[2], v[1], 0., 0., 0.],
                                [v[2], 0., -v[0], 0., 0., 0.],
                                [-v[1], v[0], 0., 0., 0., 0.],
                                [0., -v[5], v[4], 0., -v[2], v[1]],
                                [v[5], 0., -v[3], v[2], 0., -v[0]],
                                [-v[4], v[3], 0., -v[1], v[0], 0.]])
    
    res = mr.SE3.hat_adj(v)

    np.testing.assert_array_equal(res, m)
    
def test_se3_hat_adj_commute():
    v1 = np.random.rand(6)
    v2 = np.random.rand(6)
    
    res1 = mr.SE3.hat_adj(v1) @ v2
    res2 = mr.SE3.hat_commute_adj(v2) @ v1
    
    np.testing.assert_allclose(res1, res2)
    
def test_se3_exp_adj():
    v = np.random.rand(6)
    a = np.random.rand()
    res = mr.SE3.exp_adj(v, a)

    m = expm(a*mr.SE3.hat_adj(v))
    
    np.testing.assert_allclose(res, m)
    
def test_se3_exp_integ_adj():
    vec = np.random.rand(6)
    angle = np.random.rand()

    res = mr.SE3.exp_integ_adj(vec, angle)

    def integrad(s):
        return expm(s*mr.SE3.hat_adj(vec))
    
    m, _ = integrate.quad_vec(integrad, 0, angle)
        
    np.testing.assert_allclose(res, m)

def test_se3_sub_tan_vec():
    
    mat1 = mr.SE3.rand()
    mat2 = mr.SE3.rand()
    
    res = mr.SE3.sub_tan_vec(mat1, mat2, "bframe")
    sol = mr.SE3.vee(mat1.mat_inv() @ (mat2.mat() - mat1.mat()))
    
    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

    res = mr.SE3.sub_tan_vec(mat1, mat2, "fframe")
    sol = mr.SE3.vee((mat2.mat() - mat1.mat()) @ mat1.mat_inv())
    
    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

def test_se3_matmul():
    v = np.random.rand(6)
    v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
    m = mr.SO3.exp(v[0:3]) 
    
    h1 = mr.SE3(m, v[3:6])
    h2 = mr.SE3(m.transpose(), -m.transpose() @ v[3:6])
    res = h1@h2
    
    np.testing.assert_allclose(res.mat(), np.eye(4), rtol=1e-15, atol=1e-15)
    
def test_se3_matmul_mat4d():
    v = np.random.rand(6)
    v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
    m = mr.SO3.exp(v[0:3]) 
    
    mat = mr.SE3(m, v[3:6])
    
    m = mr.SE3(m.transpose(), -m.transpose() @ v[3:6]).mat()

    res = mat@m
    
    np.testing.assert_allclose(res, np.eye(4), rtol=1e-15, atol=1e-15)

def test_se3_matmul_mat6d():
    v = np.random.rand(6)
    v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
    m = mr.SO3.exp(v[0:3]) 
    
    mat = mr.SE3(m, v[3:6])
    
    m = mr.SE3(m.transpose(), -m.transpose() @ v[3:6]).mat_adj()

    res = mat@m
    
    np.testing.assert_allclose(res, np.eye(6), rtol=1e-15, atol=1e-15)

def test_se3_matmul_vec6d():
    v = np.random.rand(6)
    v[0:3] = v[0:3] / np.linalg.norm(v[0:3])
    m = mr.SO3.exp(v[0:3]) 
    
    h = mr.SE3(m, v[3:6])
    vec = np.random.rand(6)

    res = h @ vec

    ref = np.zeros(6)
    ref[0:3] = m @ vec[0:3]
    ref[3:6] = mr.SO3.hat(v[3:6]) @ m @ vec[0:3] + m @ vec[3:6]

    np.testing.assert_allclose(res, ref, rtol=1e-15, atol=1e-15)

def test_se3_matmul_vec3d():
    h = mr.SE3.rand()
    vec = np.random.rand(3)

    res = h @ vec

    ref = h.rot() @ vec + h.pos()

    np.testing.assert_allclose(res, ref, rtol=1e-15, atol=1e-15)
    
def test_se3_jac_lie_wrt_scaler():
    v = np.random.rand(6)
    dv = np.random.rand(6)
    a = np.random.rand()
    eps = 1e-8
    
    res = mr.jac_lie_wrt_scaler(mr.SE3, v, a, dv)
    
    h = mr.SE3.exp(v, a)
    v_ = v + dv*eps
    h_ = mr.SE3.exp(v_, a)
    
    dh = (h_ - h) / eps
    
    np.testing.assert_allclose(res, dh, 1e-3)
    
def test_se3_jac_lie_wrt_scaler_integ():
    v = np.random.rand(6)
    dv = np.random.rand(6)
    a = np.random.rand()
    eps = 1e-8
    
    def integrad(s):
        return mr.jac_lie_wrt_scaler(mr.SE3, v, s, dv)
    
    res, _ = integrate.quad_vec(integrad, 0, a)
    
    h = mr.SE3.exp_integ(v, a)
    v_ = v + dv*eps
    h_ = mr.SE3.exp_integ(v_, a)
    
    dh = (h_ - h) / eps
    
    np.testing.assert_allclose(res, dh, 1e-3)

def test_se3_jac_adj_lie_wrt_scaler():
    v = np.random.rand(6)
    dv = np.random.rand(6)
    a = np.random.rand()
    eps = 1e-8
    
    res = mr.jac_adj_lie_wrt_scaler(mr.SE3, v, a, dv)
    
    h = mr.SE3.exp_adj(v, a)
    v_ = v + dv*eps
    h_ = mr.SE3.exp_adj(v_, a)
    
    dh = (h_ - h) / eps
    
    np.testing.assert_allclose(res, dh, 1e-3)
    
def test_se3_adj_jac_adj_lie_wrt_scaler_integ():
    v = np.random.rand(6)
    dv = np.random.rand(6)
    a = np.random.rand()
    eps = 1e-8
    
    def integrad(s):
        return mr.jac_adj_lie_wrt_scaler(mr.SE3, v, s, dv)
    
    res, _ = integrate.quad_vec(integrad, 0, a)
    
    h = mr.SE3.exp_integ_adj(v, a)
    v_ = v + dv*eps
    h_ = mr.SE3.exp_integ_adj(v_, a)
    
    dh = (h_ - h) / eps
    
    np.testing.assert_allclose(res, dh, 1e-3)
