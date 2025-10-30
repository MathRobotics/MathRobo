import numpy as np
import jax.numpy as jnp
from typing import Union

class FactorialVector:
    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self.n = vecs.shape[0]
        self.vecs = vecs

    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self.vecs.flatten()
    
    def vec_factorial(self) -> Union[np.ndarray, jnp.ndarray]:
        result = []
        for i in range(self.n):
            fact = 1
            for j in range(1, i + 1):
                fact *= j
            result.append(fact * self.vecs[i])
        if isinstance(self.vecs, np.ndarray):
            return np.array(result)
        else:
            return jnp.array(result)