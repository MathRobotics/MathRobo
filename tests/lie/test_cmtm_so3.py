import numpy as np

import mathrobo as mr
from mathrobo.basic import basic

test_order = 10

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

    np.testing.assert_allclose(res.mat(), mat, rtol=1e-10, atol=1e-10)

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

    np.testing.assert_allclose(res.mat(), mat, rtol=1e-10, atol=1e-10)
    
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

    np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-10, atol=1e-10)
    
def test_cmtm_so3_adj_vec2d():
    so3 = mr.SO3.rand()
    vec = np.random.rand(2,3)

    res = mr.CMTM[mr.SO3](so3, vec)
    
    mat = np.zeros((9,9))
    mat[0:3,0:3] = mat[3:6,3:6] = mat[6:9,6:9] = so3.mat_adj()
    mat[3:6,0:3] = mat[6:9,3:6] = so3.mat_adj() @ so3.hat_adj(vec[0])
    mat[6:9,0:3] = so3.mat_adj() @ (so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) * 0.5

    np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-10, atol=1e-10)

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

    np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-10, atol=1e-10)
    
def test_cmtm_so3_getter():
    so3 = mr.SO3.rand() 
    vec = np.random.rand(test_order,3)

    res = mr.CMTM[mr.SO3](so3,vec)
    
    np.testing.assert_array_equal(res.elem_mat(), so3.mat())
    for i in range(test_order):
        np.testing.assert_array_equal(res.elem_vecs(i), vec[i])

def test_cmtm_so3_set_mat():
    so3 = mr.SO3.rand() 
    vec = np.random.rand(test_order,3)

    cmtm = mr.CMTM[mr.SO3](so3,vec)

    res = mr.CMTM.set_mat(mr.SO3, cmtm.mat())

    np.testing.assert_allclose(res.elem_mat(), cmtm.elem_mat(), rtol=1e-8, atol=1e-8)
    for i in range(test_order):
        np.testing.assert_allclose(res.elem_vecs(i), cmtm.elem_vecs(i), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(res.elem_vecs(i), vec[i], rtol=1e-8, atol=1e-8)

def test_cmtm_so3_vecs():
    so3 = mr.SO3.rand()
    vec = np.random.rand(test_order,3)

    res = mr.CMTM[mr.SO3](so3, vec)
    np.testing.assert_allclose(res.vecs(), vec)
    np.testing.assert_allclose(res.vecs(3), vec[:2])

def test_cmtm_so3_vecs_flatten():
    so3 = mr.SO3.rand()
    vec = np.random.rand(test_order,3)

    res = mr.CMTM[mr.SO3](so3, vec)
    np.testing.assert_array_equal(res.vecs_flatten(), vec.flatten())
    np.testing.assert_array_equal(res.vecs_flatten(3), vec[:2].flatten())

def test_cmtm_so3_inv():
    so3 = mr.SO3.rand()
    
    for i in range(test_order):
        vel = np.random.rand(i,3)

        res = mr.CMTM[mr.SO3](so3, vel)

        expected_mat = np.eye(3*(i+1))
        result_mat = res @ res.inv()

        np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-10, atol=1e-10)

        result_mat = res.inv() @ res
        np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-10, atol=1e-10)
    
def test_cmtm_so3_mat_inv():
    so3 = mr.SO3.rand()
    
    for i in range(test_order):
        vel = np.random.rand(i,3)

        res = mr.CMTM[mr.SO3](so3, vel)
        
        expected_mat = np.eye(3*(i+1))
        result_mat = res.mat() @ res.mat_inv()

        np.testing.assert_allclose(result_mat, expected_mat, rtol=1e-10, atol=1e-10)

def test_cmtm_so3_mat_inv_elem():
    so3 = mr.SO3.rand()
    res = mr.CMTM[mr.SO3](so3)

    result = res.mat_inv()
    expected = so3.mat_inv()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(1,3)
    res = mr.CMTM[mr.SO3](so3, vec)

    result = res.mat_inv()
    expected = np.zeros((6,6))
    expected[0:3,0:3] = expected[3:6,3:6] = so3.mat_inv()
    expected[3:6,0:3] = -so3.hat(vec[0]) @ so3.mat_inv()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(2,3)
    res = mr.CMTM[mr.SO3](so3, vec)

    result = res.mat_inv()
    expected = np.zeros((9,9))
    expected[0:3,0:3] = expected[3:6,3:6] = expected[6:9,6:9] = so3.mat_inv()
    expected[3:6,0:3] = expected[6:9,3:6] = -so3.hat(vec[0]) @ so3.mat_inv()
    expected[6:9,0:3] = (-so3.hat(vec[1]) + so3.hat(vec[0]) @ so3.hat(vec[0])) @ so3.mat_inv() * 0.5

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
        
def test_cmtm_so3_inv_adj():
    so3 = mr.SO3.rand() 
    
    for i in range(test_order):
        vel = np.random.rand(i,3)

        res = mr.CMTM[mr.SO3](so3, vel)
        
        mat = np.eye(3*(i+1))

        np.testing.assert_allclose(res.mat_adj() @ res.mat_inv_adj(), mat, rtol=1e-8, atol=1e-8)

def test_cmtm_so3_mat_inv_adj_elem():
    so3 = mr.SO3.rand()
    res = mr.CMTM[mr.SO3](so3)

    result = res.mat_inv_adj()
    expected = so3.mat_inv_adj()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(1,3)
    res = mr.CMTM[mr.SO3](so3, vec)

    result = res.mat_inv_adj()
    expected = np.zeros((6,6))
    expected[0:3,0:3] = expected[3:6,3:6] = so3.mat_inv_adj()
    expected[3:6,0:3] = -so3.hat_adj(vec[0]) @ so3.mat_inv_adj()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(2,3)
    res = mr.CMTM[mr.SO3](so3, vec)

    result = res.mat_inv_adj()
    expected = np.zeros((9,9))
    expected[0:3,0:3] = expected[3:6,3:6] = expected[6:9,6:9] = so3.mat_inv_adj()
    expected[3:6,0:3] = expected[6:9,3:6] = -so3.hat_adj(vec[0]) @ so3.mat_inv_adj()
    expected[6:9,0:3] = (-so3.hat_adj(vec[1]) + so3.hat_adj(vec[0]) @ so3.hat_adj(vec[0])) @ so3.mat_inv_adj() * 0.5

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

def test_cmtm_so3_hat():
    for i in range(1,test_order):
        vec = np.random.rand(i,3)

        res = mr.CMTM.hat(mr.SO3, vec)
        mat = np.zeros((3*i,3*i))
        for j in range(i):
            for k in range(j, i):
                mat[3*k:3*(k+1), 3*(k-j):3*(k-j+1)] = mr.SO3.hat(vec[j])

        np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_hat_adj():
    for i in range(1,test_order):
        vec = np.random.rand(i,3)

        res = mr.CMTM.hat_adj(mr.SO3, vec)
        mat = np.zeros((3*i,3*i))
        for j in range(i):
            for k in range(j, i):
                mat[3*k:3*(k+1), 3*(k-j):3*(k-j+1)] = mr.SO3.hat_adj(vec[j])

        np.testing.assert_array_equal(res, mat)

def test_cmtm_so3_vee():
    for i in range(1,test_order):
        vec = np.random.rand(i,3)

        mat = mr.CMTM.hat(mr.SO3, vec)
        res = mr.CMTM.vee(mr.SO3, mat)

        np.testing.assert_allclose(res, vec) 

def test_cmtm_so3_vee_adj():
    for i in range(1,test_order):
        vec = np.random.rand(i,3)

        mat = mr.CMTM.hat_adj(mr.SO3, vec)
        res = mr.CMTM.vee_adj(mr.SO3, mat)

        np.testing.assert_allclose(res, vec) 

def test_cmtm_so3_sub_vec():
    for i in range(test_order):
        mat1 = mr.CMTM.rand(mr.SO3, i+1)
        mat2 = mr.CMTM.rand(mr.SO3, i+1)

        res = mr.CMTM.sub_vec(mat1, mat2, "bframe")
        sol = np.zeros(3*(i+1))
        sol[0:3] = mr.SO3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
        if i > 0: 
            for j in range(i):
                sol[3*(j+1):3*(j+2)] = mat2._vecs[j] - mat1._vecs[j]
        
        np.testing.assert_allclose(res, sol, rtol=1e-10, atol=1e-10)

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
    for i in range(test_order):
        m1 = mr.CMTM.rand(mr.SO3, i+1)
        m2 = mr.CMTM.rand(mr.SO3, i+1)

        result_mat = m1 @ m2
        expected_mat = m1.mat() @ m2.mat()

        np.testing.assert_allclose(expected_mat, result_mat.mat(), rtol=1e-10, atol=1e-10)

def test_cmtm_so3_multiply_adj():
    for i in range(test_order):
        m1 = mr.CMTM.rand(mr.SO3, i+1)
        m2 = mr.CMTM.rand(mr.SO3, i+1)

        result_mat = m1 @ m2

        expected_mat = m1.mat_adj() @ m2.mat_adj()

        np.testing.assert_allclose(expected_mat, result_mat.mat_adj(), rtol=1e-10, atol=1e-10)

def test_cmtm_so3_tangent_mat():
    res = mr.CMTM.rand(mr.SO3, test_order)

    np.testing.assert_allclose(res.tangent_mat()@res.tangent_mat_inv(), np.eye(res._mat_adj_size*test_order), rtol=1e-10, atol=1e-10)

def test_cmtm_so3_tangent_mat_cm():
    res = mr.CMTM.rand(mr.SO3, test_order)

    np.testing.assert_allclose(res.tangent_mat_cm()@res.tangent_mat_cm_inv(), np.eye(res._mat_adj_size*test_order), rtol=1e-10, atol=1e-10)