import math
from typing import Union

import numpy as np
import jax.numpy as jnp

class FactorialVector:
    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self.n = vecs.shape[0]
        self.dim = vecs.shape[1] if len(vecs.shape) > 1 else 1
        self.len = vecs.flatten().shape[0]
        self.vecs = vecs
        self.factorial_mat = self._compute_factorial_matrix()

    def _compute_factorial_matrix(self) -> Union[np.ndarray, jnp.ndarray]:
        if isinstance(self.vecs, np.ndarray):
            mat = np.eye(self.len, dtype=self.vecs.dtype)
        else:
            mat = jnp.eye(self.len, dtype=self.vecs.dtype)

        for i in range(self.n):
            mat[i*self.dim:(i+1)*self.dim, i*self.dim:(i+1)*self.dim] *= math.factorial(i)
        return mat

    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self.vecs.flatten()
    
    def vec_factorial(self) -> Union[np.ndarray, jnp.ndarray]:
        return self.factorial_mat @ self.vecs.flatten()