import math
from typing import Union

import numpy as np
import jax.numpy as jnp

from .factorial_vec import *

class CMVector:
    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self._factorial_vector = FactorialVector(vecs)
        self._n = self._factorial_vector._n
        self._dim = self._factorial_vector._dim
        self._len = self._factorial_vector._len

    def vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.vecs()
    
    def cm_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.ifac_vecs()
    
    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.vec()
    
    def cm_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_vector.ifac_vec()    