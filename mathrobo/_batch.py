import numpy as np
import jax.numpy as jnp


def array_lib(LIB: str):
    if LIB == "jax":
        return jnp
    if LIB == "numpy":
        return np
    raise ValueError("Unsupported library. Choose 'numpy' or 'jax'.")


def transpose_last(mat):
    return mat.swapaxes(-1, -2)


def matvec(mat, vec):
    return (mat @ vec[..., None])[..., 0]


def flatten_last2(arr):
    return arr.reshape(arr.shape[:-2] + (arr.shape[-2] * arr.shape[-1],))
