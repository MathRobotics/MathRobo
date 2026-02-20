import math
from typing import Union

import numpy as np
import jax.numpy as jnp

class Factorial:
    @staticmethod
    def _factorial_scales(
        n: int,
        LIB: str = "numpy",
        dtype: Union[np.dtype, jnp.dtype, None] = None,
    ) -> Union[np.ndarray, jnp.ndarray]:
        if LIB == "jax":
            if dtype is None:
                dtype = jnp.float32
            return jnp.asarray([math.factorial(i) for i in range(n)], dtype=dtype)
        if dtype is None:
            dtype = np.float64
        return np.asarray([math.factorial(i) for i in range(n)], dtype=dtype)

    @classmethod
    def mat(
        cls,
        n: int,
        dim: int,
        LIB: str = "numpy",
        dtype: Union[np.dtype, jnp.dtype, None] = None,
    ) -> Union[np.ndarray, jnp.ndarray]:
        scales = cls._factorial_scales(n, LIB=LIB, dtype=dtype)
        if LIB == "jax":
            return jnp.diag(jnp.repeat(scales, dim))
        return np.diag(np.repeat(scales, dim))
    
    @classmethod
    def mat_inv(
        cls,
        n: int,
        dim: int,
        LIB: str = "numpy",
        dtype: Union[np.dtype, jnp.dtype, None] = None,
    ) -> Union[np.ndarray, jnp.ndarray]:
        scales = cls._factorial_scales(n, LIB=LIB, dtype=dtype)
        inv_scales = 1.0 / scales
        if LIB == "jax":
            return jnp.diag(jnp.repeat(inv_scales, dim))
        return np.diag(np.repeat(inv_scales, dim))

class FactorialVector:
    @staticmethod
    def _detect_lib(vecs: Union[np.ndarray, jnp.ndarray]) -> str:
        module = type(vecs).__module__
        if module.startswith("jax") or module.startswith("jaxlib"):
            return "jax"
        return "numpy"

    @staticmethod
    def _scale_dtype(vecs: Union[np.ndarray, jnp.ndarray], LIB: str):
        if LIB == "jax":
            return jnp.result_type(vecs.dtype, jnp.float32)
        return np.result_type(vecs.dtype, np.float64)

    def __init__(self, vecs : Union[np.ndarray, jnp.ndarray]):
        self._n = vecs.shape[0]
        self._dim = vecs.shape[1] if len(vecs.shape) > 1 else 1
        self._len = vecs.flatten().shape[0]
        self._vecs = vecs
        self._lib = self._detect_lib(vecs)

        dtype = self._scale_dtype(vecs, self._lib)
        self._factorial_scales = Factorial._factorial_scales(self._n, LIB=self._lib, dtype=dtype)
        self._inverse_factorial_scales = 1.0 / self._factorial_scales

        vecs_2d = vecs.reshape(self._n, self._dim)
        self._factorial_vecs = vecs_2d * self._factorial_scales.reshape(self._n, 1)
        self._inverse_factorial_vecs = vecs_2d * self._inverse_factorial_scales.reshape(self._n, 1)

        self._factorial_mat = None
        self._inverse_factorial_mat = None

    @staticmethod
    def set_fac_vecs(fac_vecs : Union[np.ndarray, jnp.ndarray]) -> 'FactorialVector':
        n = fac_vecs.shape[0]
        dim = fac_vecs.shape[1] if len(fac_vecs.shape) > 1 else 1
        lib = FactorialVector._detect_lib(fac_vecs)
        dtype = FactorialVector._scale_dtype(fac_vecs, lib)
        inv_scales = 1.0 / Factorial._factorial_scales(n, LIB=lib, dtype=dtype)
        vecs = fac_vecs.reshape(n, dim) * inv_scales.reshape(n, 1)
        return FactorialVector(vecs)
    
    @staticmethod
    def set_ifac_vecs(ifac_vecs : Union[np.ndarray, jnp.ndarray]) -> 'FactorialVector':
        n = ifac_vecs.shape[0]
        dim = ifac_vecs.shape[1] if len(ifac_vecs.shape) > 1 else 1
        lib = FactorialVector._detect_lib(ifac_vecs)
        dtype = FactorialVector._scale_dtype(ifac_vecs, lib)
        scales = Factorial._factorial_scales(n, LIB=lib, dtype=dtype)
        vecs = ifac_vecs.reshape(n, dim) * scales.reshape(n, 1)
        return FactorialVector(vecs)

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
        if self._factorial_mat is None:
            if self._lib == "jax":
                self._factorial_mat = jnp.diag(jnp.repeat(self._factorial_scales, self._dim))
            else:
                self._factorial_mat = np.diag(np.repeat(self._factorial_scales, self._dim))
        return self._factorial_mat

    def ifac_mat(self) -> Union[np.ndarray, jnp.ndarray]:
        if self._inverse_factorial_mat is None:
            if self._lib == "jax":
                self._inverse_factorial_mat = jnp.diag(jnp.repeat(self._inverse_factorial_scales, self._dim))
            else:
                self._inverse_factorial_mat = np.diag(np.repeat(self._inverse_factorial_scales, self._dim))
        return self._inverse_factorial_mat
