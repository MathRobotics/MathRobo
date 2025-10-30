import math
from typing import Union

import numpy as np
import jax.numpy as jnp

class CMVector:
    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self._factorial_vector = fv.FactorialVector(vecs)

    def vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.vecs()
    
    def cm_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.ifac_vecs()
    
    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.vec()
    
    def cm_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.ifac_vec()    