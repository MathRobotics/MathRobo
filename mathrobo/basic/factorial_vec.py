import math
from typing import Union

import numpy as np
import jax.numpy as jnp

from .._batch import flatten_last2

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
    def mat_inv(cls, n: int, dim: int) -> Union[np.ndarray, jnp.ndarray]:
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
        if vecs.ndim == 1:
            vecs = vecs[..., None]
            self._squeeze_vec = True
        else:
            self._squeeze_vec = False

        xp = jnp if isinstance(vecs, jnp.ndarray) else np
        self._n = vecs.shape[-2]
        self._dim = vecs.shape[-1]
        self._len = self._n * self._dim
        self._vecs = vecs
        self._factorial_mat = Factorial.mat(self._n, self._dim)
        self._inverse_factorial_mat = Factorial.mat_inv(self._n, self._dim)
        factors = xp.asarray([math.factorial(i) for i in range(self._n)], dtype=vecs.dtype)
        factors = factors.reshape((1,) * (vecs.ndim - 2) + (self._n, 1))
        self._factorial_vecs = vecs * factors
        self._inverse_factorial_vecs = vecs / factors

    @staticmethod
    def set_fac_vecs(fac_vecs : Union[np.ndarray, jnp.ndarray]) -> 'FactorialVector':
        if fac_vecs.ndim == 1:
            fac_vecs = fac_vecs[..., None]
        xp = jnp if isinstance(fac_vecs, jnp.ndarray) else np
        n = fac_vecs.shape[-2]
        factors = xp.asarray([math.factorial(i) for i in range(n)], dtype=fac_vecs.dtype)
        factors = factors.reshape((1,) * (fac_vecs.ndim - 2) + (n, 1))
        vecs = fac_vecs / factors
        return FactorialVector(vecs)
    
    @staticmethod
    def set_ifac_vecs(ifac_vecs : Union[np.ndarray, jnp.ndarray]) -> 'FactorialVector':
        if ifac_vecs.ndim == 1:
            ifac_vecs = ifac_vecs[..., None]
        xp = jnp if isinstance(ifac_vecs, jnp.ndarray) else np
        n = ifac_vecs.shape[-2]
        factors = xp.asarray([math.factorial(i) for i in range(n)], dtype=ifac_vecs.dtype)
        factors = factors.reshape((1,) * (ifac_vecs.ndim - 2) + (n, 1))
        vecs = ifac_vecs * factors
        return FactorialVector(vecs)

    def vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._squeeze_vec:
            return self._vecs[..., 0]
        return self._vecs

    def fac_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._squeeze_vec:
            return self._factorial_vecs[..., 0]
        return self._factorial_vecs
    
    def ifac_vecs(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._squeeze_vec:
            return self._inverse_factorial_vecs[..., 0]
        return self._inverse_factorial_vecs

    def vec(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._squeeze_vec and self._vecs.ndim == 2:
            return self._vecs[..., 0]
        return flatten_last2(self._vecs)

    def fac_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._squeeze_vec and self._factorial_vecs.ndim == 2:
            return self._factorial_vecs[..., 0]
        return flatten_last2(self._factorial_vecs)
    
    def ifac_vec(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._squeeze_vec and self._inverse_factorial_vecs.ndim == 2:
            return self._inverse_factorial_vecs[..., 0]
        return flatten_last2(self._inverse_factorial_vecs)
    
    def fac_mat(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._factorial_mat

    def ifac_mat(self) -> Union[np.ndarray, jnp.ndarray]:
        return self._inverse_factorial_mat
