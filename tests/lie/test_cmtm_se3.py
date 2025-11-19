import numpy as np
import math

import mathrobo as mr

test_order = 10

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

    np.testing.assert_allclose(res.mat(), mat, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_vec3d():
    se3 = mr.SE3.rand()  
    vec = np.random.rand(3,6)

    res = mr.CMTM[mr.SE3](se3, vec)
    
    mat = np.zeros((16,16))
    mat[0:4,0:4] = mat[4:8,4:8] = mat[8:12,8:12] = mat[12:16,12:16] = se3.mat()
    mat[4:8,0:4] = mat[8:12,4:8] = mat[12:16,8:12] = se3.mat() @ se3.hat(vec[0])
    mat[8:12,0:4] = mat[12:16,4:8] =  se3.mat() @ (se3.hat(vec[1]) + se3.hat(vec[0]) @ se3.hat(vec[0])) * 0.5
    tmp1 = se3.mat() @ se3.hat(vec[2]/2)
    tmp2 = se3.mat() @ se3.hat(vec[0]) @ se3.hat(vec[1])
    tmp3 = se3.mat() @ (se3.hat(vec[1]) + se3.hat(vec[0]) @ se3.hat(vec[0])) * 0.5 @ se3.hat(vec[0])
    mat[12:16,0:4] = ( tmp1 + tmp2 + tmp3 ) / 3.0

    np.testing.assert_allclose(res.mat(), mat, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_adj():
    se3 = mr.SE3.rand()
    res = mr.CMTM[mr.SE3](se3)

    np.testing.assert_allclose(res.mat_adj(), se3.mat_adj(), rtol=1e-10, atol=1e-10)
    
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

    np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_adj_vec3d():
    se3 = mr.SE3.rand()
    vec = np.random.rand(3,6)

    res = mr.CMTM[mr.SE3](se3, vec)
    
    mat = np.zeros((24,24))
    mat[0:6,0:6] = mat[6:12,6:12] = mat[12:18,12:18] = mat[18:24,18:24] = se3.mat_adj()
    mat[6:12,0:6] = mat[12:18,6:12] = mat[18:24,12:18] = se3.mat_adj() @ se3.hat_adj(vec[0])
    mat[12:18,0:6] = mat[18:24,6:12] = se3.mat_adj() @ (se3.hat_adj(vec[1]) + se3.hat_adj(vec[0]) @ se3.hat_adj(vec[0])) * 0.5
    tmp1 = se3.mat_adj() @ se3.hat_adj(vec[2]/2)
    tmp2 = se3.mat_adj() @ se3.hat_adj(vec[0]) @ se3.hat_adj(vec[1])
    tmp3 = se3.mat_adj() @ (se3.hat_adj(vec[1]) + se3.hat_adj(vec[0]) @ se3.hat_adj(vec[0])) * 0.5 @ se3.hat_adj(vec[0])
    mat[18:24,0:6] = ( tmp1 + tmp2 + tmp3 ) / 3.0

    np.testing.assert_allclose(res.mat_adj(), mat, rtol=1e-10, atol=1e-10)
    
def test_cmtm_se3_getter():
    se3 = mr.SE3.rand()  
    vec = np.random.rand(test_order,6)

    res = mr.CMTM[mr.SE3](se3,vec)
    
    np.testing.assert_array_equal(res.elem_mat(), se3.mat())
    for i in range(test_order):
        np.testing.assert_array_equal(res.elem_vecs(i), vec[i])

def test_cmtm_se3_set_mat():
    se3 = mr.SE3.rand() 
    vec = np.random.rand(test_order,6)

    cmtm = mr.CMTM[mr.SE3](se3,vec)

    res = mr.CMTM.set_mat(mr.SE3, cmtm.mat())

    np.testing.assert_allclose(res.elem_mat(), cmtm.elem_mat(), rtol=1e-8, atol=1e-8)
    for i in range(test_order):
        np.testing.assert_allclose(res.elem_vecs(i), cmtm.elem_vecs(i), rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(res.elem_vecs(i), vec[i], rtol=1e-8, atol=1e-8)

def test_cmtm_se3_vecs():
    se3 = mr.SE3.rand()
    vec = np.random.rand(test_order,6)

    res = mr.CMTM[mr.SE3](se3, vec)
    np.testing.assert_array_equal(res.vecs(), vec)
    np.testing.assert_array_equal(res.vecs(3), vec[:2])

def test_cmtm_se3_vecs_flatten():
    se3 = mr.SE3.rand()
    vec = np.random.rand(test_order,6)

    res = mr.CMTM[mr.SE3](se3, vec)
    np.testing.assert_array_equal(res.vecs_flatten(), vec.flatten())
    np.testing.assert_array_equal(res.vecs_flatten(3), vec[:2].flatten())

def test_cmtm_se3_inv():
    for i in range(4):
        res = mr.CMTM.rand(mr.SE3,i+1)

        expected_mat = np.eye(4*(i+1))
        result_mat = res @ res.inv()
        
        np.testing.assert_allclose(result_mat.mat(), expected_mat, rtol=1e-10, atol=1e-10)
    
def test_cmtm_se3_mat_inv():
    se3 = mr.SE3.rand()  

    for i in range(5):
        vel = np.random.rand(i,6)
        
        res = mr.CMTM[mr.SE3](se3, vel)
        
        mat = np.eye(4*(i+1))

        np.testing.assert_allclose(res.mat() @ res.mat_inv(), mat, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_mat_inv_elem():
    se3 = mr.SE3.rand()
    res = mr.CMTM[mr.SE3](se3)

    result = res.mat_inv()
    expected = se3.mat_inv()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(1,6)
    res = mr.CMTM[mr.SO3](se3, vec)

    result = res.mat_inv()
    expected = np.zeros((8,8))
    expected[0:4,0:4] = expected[4:8,4:8] = se3.mat_inv()
    expected[4:8,0:4] = -se3.hat(vec[0]) @ se3.mat_inv()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(2,6)
    res = mr.CMTM[mr.SO3](se3, vec)

    result = res.mat_inv()
    expected = np.zeros((12,12))
    expected[0:4,0:4] = expected[4:8,4:8] = expected[8:12,8:12] = se3.mat_inv()
    expected[4:8,0:4] = expected[8:12,4:8] = -se3.hat(vec[0]) @ se3.mat_inv()
    expected[8:12,0:4] = (-se3.hat(vec[1]) + se3.hat(vec[0]) @ se3.hat(vec[0])) @ se3.mat_inv() * 0.5

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
        
def test_cmtm_se3_inv_adj():
    se3 = mr.SE3.rand()   
    
    for i in range(test_order):
        vel = np.random.rand(i,6)

        res = mr.CMTM[mr.SE3](se3, vel)
        
        mat = np.eye(6*(i+1))
        
        np.testing.assert_allclose(res.mat_adj() @ res.mat_inv_adj(), mat, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_mat_inv_adj_elem():
    se3 = mr.SE3.rand()
    res = mr.CMTM[mr.SE3](se3)

    result = res.mat_inv_adj()
    expected = se3.mat_inv_adj()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(1,6)
    res = mr.CMTM[mr.SE3](se3, vec)

    result = res.mat_inv_adj()
    expected = np.zeros((12,12))
    expected[0:6,0:6] = expected[6:12,6:12] = se3.mat_inv_adj()
    expected[6:12,0:6] = -se3.hat_adj(vec[0]) @ se3.mat_inv_adj()

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    vec = np.random.rand(2,6)
    res = mr.CMTM[mr.SE3](se3, vec)

    result = res.mat_inv_adj()
    expected = np.zeros((18,18))
    expected[0:6,0:6] = expected[6:12,6:12] = expected[12:18,12:18] = se3.mat_inv_adj()
    expected[6:12,0:6] = expected[12:18,6:12] = -se3.hat_adj(vec[0]) @ se3.mat_inv_adj()
    expected[12:18,0:6] = (-se3.hat_adj(vec[1]) + se3.hat_adj(vec[0]) @ se3.hat_adj(vec[0])) @ se3.mat_inv_adj() * 0.5

    np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_hat():
    for i in range(1,test_order+1):
        vec = np.random.rand(i,6)

        res = mr.CMTM.hat(mr.SE3, vec)
        mat = np.zeros((4*i,4*i))
        for j in range(i):
            for k in range(j, i):
                mat[4*k:4*(k+1), 4*(k-j):4*(k-j+1)] = mr.SE3.hat(vec[j])
        np.testing.assert_array_equal(res, mat)

def test_cmtm_se3_hat_adj():
    for i in range(1,test_order):
        vec = np.random.rand(i,6)

        res = mr.CMTM.hat_adj(mr.SE3, vec)
        mat = np.zeros((6*i,6*i))
        for j in range(i):
            for k in range(j, i):
                mat[6*k:6*(k+1), 6*(k-j):6*(k-j+1)] = mr.SE3.hat_adj(vec[j])
        np.testing.assert_array_equal(res, mat)

def test_cmtm_se3_vee():
    for i in range(1,test_order):
        vec = np.random.rand(i,6)

        mat = mr.CMTM.hat(mr.SE3, vec)
        res = mr.CMTM.vee(mr.SE3, mat)

        np.testing.assert_allclose(res, vec) 

def test_cmtm_se3_vee_adj():
    for i in range(1,test_order):
        vec = np.random.rand(i,6)

        mat = mr.CMTM.hat_adj(mr.SE3, vec)
        res = mr.CMTM.vee_adj(mr.SE3, mat)

        np.testing.assert_allclose(res, vec) 

def test_cmtm_se3_hat_commute_adj():
    for i in range(1,test_order):
        vec1 = np.random.rand(i,6)
        vec2 = np.random.rand(i,6)

        res = mr.CMTM.hat_adj(mr.SE3, vec1) @ vec2.flatten()
        ans = mr.CMTM.hat_commute_adj(mr.SE3, vec2) @ vec1.flatten()    

        np.testing.assert_allclose(res, ans)

        res = mr.CMTM.hat_adj(mr.SE3wrench, vec1) @ vec2.flatten()
        ans = mr.CMTM.hat_commute_adj(mr.SE3wrench, vec2) @ vec1.flatten()    

        np.testing.assert_allclose(res, ans)

def test_cmtm_se3_sub_vec():
    for i in range(test_order):
        mat1 = mr.CMTM.rand(mr.SE3, i+1)
        mat2 = mr.CMTM.rand(mr.SE3, i+1)

        res = mr.CMTM.sub_vec(mat1, mat2, "bframe")
        sol = np.zeros(6*(i+1))
        sol[0:6] = mr.SE3.sub_tan_vec(mat1._mat, mat2._mat, "bframe")
        if i > 0: 
            for j in range(i):
                sol[6*(j+1):6*(j+2)] = mat2._vecs[j] - mat1._vecs[j]
        
        np.testing.assert_allclose(res, sol, rtol=1e-10, atol=1e-10)

def test_cmtm_se3_matmul():
    order = 4
    m1 = mr.CMTM.rand(mr.SE3, order)
    m2 = mr.CMTM.rand(mr.SE3, order)

    result = m1 @ m2

    expected_frame = m1.elem_mat() @ m2.elem_mat()
    expected_veloc = m2._mat.mat_inv_adj() @ m1.elem_vecs(0) + m2.elem_vecs(0)
    expected_accel = \
        m2._mat.mat_inv_adj() @ m1.elem_vecs(1) +\
        mr.SE3.hat_adj( m2._mat.mat_inv_adj() @ m1.elem_vecs(0) ) @ m2.elem_vecs(0) +\
        m2.elem_vecs(1)
    expected_jerk = \
        m2._mat.mat_inv_adj() @ m1.elem_vecs(2) +\
        - 2* mr.SE3.hat_adj( m2.elem_vecs(0)) @ m2._mat.mat_inv_adj() @ m1.elem_vecs(1) \
        + (-mr.SE3.hat_adj(m2.elem_vecs(1)) + mr.SE3.hat_adj( m2.elem_vecs(0)) @ mr.SE3.hat_adj( m2.elem_vecs(0))) @ m2._mat.mat_inv_adj() @ m1.elem_vecs(0) \
        + m2.elem_vecs(2)
        
    assert np.allclose(result.elem_mat(), expected_frame)
    assert np.allclose(result.elem_vecs(0), expected_veloc)
    assert np.allclose(result.elem_vecs(1), expected_accel)
    assert np.allclose(result.elem_vecs(2), expected_jerk)

def test_cmtm_se3_multiply():
    for i in range(test_order):
        m1 = mr.CMTM.rand(mr.SE3, i+1)
        m2 = mr.CMTM.rand(mr.SE3, i+1)

        result_mat = m1 @ m2
        expected_mat = m1.mat() @ m2.mat()

        np.testing.assert_allclose(expected_mat, result_mat.mat(), rtol=1e-10, atol=1e-10)

def test_cmtm_se3_multiply_adj():
    for i in range(test_order):
        m1 = mr.CMTM.rand(mr.SE3, i+1)
        m2 = mr.CMTM.rand(mr.SE3, i+1)

        result_mat = m1 @ m2

        expected_mat = m1.mat_adj() @ m2.mat_adj()

        np.testing.assert_allclose(expected_mat, result_mat.mat_adj(), rtol=1e-10, atol=1e-10)

def test_cmtm_se3_tangent_mat():
    res = mr.CMTM.rand(mr.SE3, test_order)

    np.testing.assert_allclose(res.tangent_mat()@res.tangent_mat_inv(), np.eye(res._mat_adj_size*test_order), rtol=1e-10, atol=1e-10)

def test_cmtm_se3_tangent_mat_cm():
    res = mr.CMTM.rand(mr.SE3, test_order)

    np.testing.assert_allclose(res.tangent_mat_cm()@res.tangent_mat_cm_inv(), np.eye(res._mat_adj_size*test_order), rtol=1e-10, atol=1e-10)

def test_cmtm_se3_change_elemclass():

    res = mr.CMTM.rand(mr.SE3, test_order)
    res2 = mr.CMTM.change_elemclass(res, mr.SE3)

    np.testing.assert_allclose(res.mat_adj(), res2.mat_adj(), rtol=1e-10, atol=1e-10)

    res = mr.CMTM.rand(mr.SE3, 1)
    res2 = mr.CMTM.change_elemclass(res, mr.SE3wrench)

    np.testing.assert_allclose(res.mat_inv_adj().T, res2.mat_adj(), rtol=1e-10, atol=1e-10)

def test_cmtm_se3_mat_var_x_arb_vec():
    mat = mr.CMTM.rand(mr.SE3, test_order)
    arb_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))
    tan_var_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))

    res = mat.mat_var_x_arb_vec(arb_vec, tan_var_vec, frame='bframe').cm_vec()
    sol = mat.mat_adj() @ mr.CMTM.hat_cm_commute_adj(mr.SE3, arb_vec) @ tan_var_vec.cm_vec()

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

    res = mat.mat_var_x_arb_vec(arb_vec, tan_var_vec, frame='fframe').cm_vec()
    sol = mr.CMTM.hat_cm_commute_adj(mr.SE3, mat @ arb_vec) @ tan_var_vec.cm_vec()

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

def test_cmtm_se3_mat_var_x_arb_vec_jacob():
    mat = mr.CMTM.rand(mr.SE3, test_order)
    arb_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))

    res = mat.mat_var_x_arb_vec_jacob(arb_vec, frame='bframe')
    sol = mat.mat_adj() @ mr.CMTM.hat_cm_commute_adj(mr.SE3, arb_vec)

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

    res = mat.mat_var_x_arb_vec_jacob(arb_vec, frame='fframe')
    sol = mr.CMTM.hat_cm_commute_adj(mr.SE3, mat @ arb_vec)

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

    mat_se3_cm = mr.CMTM.rand(mr.SE3, 1)
    mat_se3 = mr.SE3.set_mat(mat_se3_cm.elem_mat())
    arb_vec = np.random.rand(1,mat_se3_cm.adj_size())
    np.testing.assert_allclose(
        mat_se3_cm.mat_var_x_arb_vec_jacob(mr.cmvec.CMVector(arb_vec), frame='bframe'),
        mat_se3.mat_var_x_arb_vec_jacob(arb_vec[0], frame='bframe'),
        rtol=1e-15, atol=1e-15
    )

def test_cmtm_se3_mat_var_x_arb_vec_num_jacob():
    mat = mr.CMTM.rand(mr.SE3, test_order)
    arb_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))

    def func(dvec):
        '''
            dX = X @ hat(dvec)
            return (X + dX) @ arb_vec
        '''
        v = mr.cmvec.CMVector.set_cmvecs(dvec.reshape(mat._n, -1))
        return mat.mat_adj() @ (np.eye(mat.adj_size()) + mr.CMTM.hat_cm_adj(mr.SE3, v)) @ arb_vec.cm_vec()

    res = mat.mat_var_x_arb_vec_jacob(arb_vec, frame='bframe')
    jacob_num = mr.numerical_grad(np.zeros(mat.adj_size()), func)

    np.testing.assert_allclose(res, jacob_num, rtol=1e-6, atol=1e-6)

def test_cmtm_se3_wrench_mat_var_x_arb_vec():
    mat = mr.CMTM.rand(mr.SE3wrench, test_order)
    arb_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))
    tan_var_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))

    res = mat.mat_var_x_arb_vec(arb_vec, tan_var_vec, frame='bframe').cm_vec()
    sol = mat.mat_adj() @ mr.CMTM.hat_cm_commute_adj(mr.SE3wrench, arb_vec) @ tan_var_vec.cm_vec()

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

    res = mat.mat_var_x_arb_vec(arb_vec, tan_var_vec, frame='fframe').cm_vec()
    sol = mr.CMTM.hat_cm_commute_adj(mr.SE3wrench, mat @ arb_vec) @ tan_var_vec.cm_vec()

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

def test_cmtm_se3_wrench_mat_var_x_arb_vec_jacob():
    mat = mr.CMTM.rand(mr.SE3wrench, test_order)
    arb_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(test_order, -1))

    res = mat.mat_var_x_arb_vec_jacob(arb_vec, frame='bframe')
    sol = mat.mat_adj() @ mr.CMTM.hat_cm_commute_adj(mr.SE3wrench, arb_vec)

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

    res = mat.mat_var_x_arb_vec_jacob(arb_vec, frame='fframe')
    sol = mr.CMTM.hat_cm_commute_adj(mr.SE3wrench, mat @ arb_vec)

    np.testing.assert_allclose(res, sol, rtol=1e-15, atol=1e-15)

def test_cmtm_se3_wrench_mat_var_x_arb_vec_num_jacob():
    order = 1
    mat = mr.CMTM.rand(mr.SE3wrench, order)
    arb_vec = mr.cmvec.CMVector(np.random.rand(mat.adj_size()).reshape(order, -1))

    def func(dvec):
        '''
            dX = X @ hat(dvec)
            return (X + dX) @ arb_vec
        '''
        v = mr.cmvec.CMVector.set_cmvecs(dvec.reshape(mat._n, -1))
        return mat.mat_adj() @ (np.eye(mat.adj_size()) + mr.CMTM.hat_cm_adj(mr.SE3wrench, v)) @ arb_vec.cm_vec()

    res = mat.mat_var_x_arb_vec_jacob(arb_vec, frame='bframe')
    jacob_num = mr.numerical_grad(np.zeros(mat.adj_size()), func)

    np.testing.assert_allclose(res, jacob_num, rtol=1e-6, atol=1e-6)