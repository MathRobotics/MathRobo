import jax.numpy as jnp
from jax import jacrev
import numpy as np

from scipy.linalg import expm
from scipy import integrate

import mathrobo as mr

def test_so3_grad():
    vec = jnp.array([1.5, 1, 0])
    a = 2.0

    def func(x):
        return mr.SO3.exp(vec, x, 'jax')

    # Compute the gradient using JAX's automatic differentiation
    grad_func = jacrev(func)
    grad_result = grad_func(a)

    # Verify the result
    expected = mr.SO3.hat(vec, 'jax') @ mr.SO3.exp(vec, a, 'jax')
    
    np.testing.assert_allclose(grad_result, expected, rtol=1e-6, atol=1e-6)
