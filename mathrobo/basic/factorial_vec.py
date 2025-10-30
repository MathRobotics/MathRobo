import math
from typing import Union

import numpy as np
import jax.numpy as jnp

class Factorial:
    @classmethod
    def mat(cls, n: int, dim: int) -> Union[np.ndarray, jnp.ndarray]:
        length = n * dim
        if isinstance(dim, int):
            mat = np.eye(length)
        else:
            mat = jnp.eye(length)

        for i in range(n):
            mat[i*dim:(i+1)*dim, i*dim:(i+1)*dim] *= math.factorial(i)
        return mat
    
    @classmethod
    def inverse_mat(cls, n: int, dim: int) -> Union[np.ndarray, jnp.ndarray]:
        length = n * dim
        if isinstance(dim, int):
            mat = np.eye(length)
        else:
            mat = jnp.eye(length)

        for i in range(n):
            mat[i*dim:(i+1)*dim, i*dim:(i+1)*dim] /= math.factorial(i)
        return mat
class FactorialVector:
    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self._n = vecs.shape[0]
        self._dim = vecs.shape[1] if len(vecs.shape) > 1 else 1
        self._len = vecs.flatten().shape[0]
        self._vecs = vecs
        self._factorial_mat = Factorial.mat(self._n, self._dim)
        self._inverse_factorial_mat = Factorial.inverse_mat(self._n, self._dim)
        self._factorial_vecs = (self._factorial_mat @ vecs.flatten()).reshape(self._n, self._dim)
        self._inverse_factorial_vecs = (self._inverse_factorial_mat @ vecs.flatten()).reshape(self._n, self._dim)
    
    def vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._vecs

    def fac_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vecs
    
    def ifac_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._inverse_factorial_vecs

    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._vecs.flatten()

    def fac_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vecs.flatten()
    
    def ifac_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._inverse_factorial_vecs.flatten()
    
    def fac_mat(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_mat

    def ifac_mat(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._inverse_factorial_mat