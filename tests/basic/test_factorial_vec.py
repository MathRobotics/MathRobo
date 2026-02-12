import numpy as np
import math

import mathrobo as mr

test_order = 10
test_dim = 6

def test_factorial_vector():
    vecs = np.random.rand(test_order,test_dim)
    factorial_vector = mr.FactorialVector(vecs)
    assert factorial_vector.vecs().shape == (test_order,test_dim)
    assert factorial_vector.vec().shape == (test_dim*test_order,)
    assert factorial_vector.fac_vec().shape == (test_dim*test_order,)

    np.testing.assert_allclose(factorial_vector.vecs(), vecs)
    for i in range(test_order):
        np.testing.assert_allclose(factorial_vector.vecs()[i], vecs[i])
    for i in range(test_order):
        np.testing.assert_allclose(factorial_vector.vecs()[i:], vecs[i:])
    np.testing.assert_allclose(factorial_vector.vec(), vecs.flatten())

    expected = np.array([math.factorial(i) * vecs[i] for i in range(test_order)])
    np.testing.assert_allclose(factorial_vector.fac_vec(), expected.flatten())

def test_factorial_matrix():
    vec = np.random.rand(test_order)
    factorial_vector = mr.FactorialVector(vec)
    factorial_mat = factorial_vector.fac_mat()
    inverse_factorial_mat = factorial_vector.ifac_mat()

    identity_mat = factorial_mat @ inverse_factorial_mat
    np.testing.assert_allclose(identity_mat, np.eye(test_order))