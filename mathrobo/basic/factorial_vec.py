import math
from typing import Union

import numpy as np
import jax.numpy as jnp

class FactorialVector:
    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self._n = vecs.shape[0]
        self._dim = vecs.shape[1] if len(vecs.shape) > 1 else 1
        self._len = vecs.flatten().shape[0]
        self._vecs = vecs
        self.factorial_mat = self._compute_factorial_matrix()

    def _compute_factorial_matrix(self) -> Union[np.ndarray, jnp.ndarray]:
        if isinstance(self._vecs, np.ndarray):
            mat = np.eye(self._len, dtype=self._vecs.dtype)
        else:
            mat = jnp.eye(self._len, dtype=self._vecs.dtype)

        for i in range(self._n):
            mat[i*self._dim:(i+1)*self._dim, i*self._dim:(i+1)*self._dim] *= math.factorial(i)
        return mat

    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._vecs.flatten()

    def vec_factorial(self) -> Union[np.ndarray, jnp.ndarray]:
        return self.factorial_mat @ self._vecs.flatten()
    
    def factorial_mat(self) -> Union[np.ndarray, jnp.ndarray]:
        return self.factorial_mat