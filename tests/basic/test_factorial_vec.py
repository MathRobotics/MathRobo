import numpy as np
import math

import mathrobo as mr

test_order = 10

def test_factorial_vector():
    vec = np.random.rand(test_order)
    factorial_vector = mr.FactorialVector(vec)
    assert factorial_vector.vec().shape == (test_order,)
    assert factorial_vector.vec_factorial().shape == (test_order,)

    expected = np.array([math.factorial(i) * vec[i] for i in range(test_order)])
    np.testing.assert_allclose(factorial_vector.vec_factorial(), expected)