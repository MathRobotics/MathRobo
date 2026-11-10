"""User-defined positive affine transformations of the line: x -> scale*x + offset.

Run from an environment containing MathRobo: python examples/custom_lie_group.py
No MathRobo source changes, registry or private imports are required.
"""
import numpy as np
import jax.numpy as jnp
from mathrobo import MatrixLieGroup, CMTM


class Affine1D(MatrixLieGroup):
    @staticmethod
    def dof():
        return 2

    @staticmethod
    def mat_size():
        return 2

    @staticmethod
    def hat(vec, LIB='numpy'):
        xp = jnp if LIB == 'jax' else np
        vec = xp.asarray(vec)
        if vec.ndim < 1 or vec.shape[-1] != 2:
            raise ValueError('Expected (..., 2) affine tangent coordinates')
        zero = xp.zeros_like(vec)
        return xp.stack((vec, zero), axis=-2)

    @staticmethod
    def vee(matrix, LIB='numpy'):
        xp = jnp if LIB == 'jax' else np
        matrix = xp.asarray(matrix)
        if matrix.ndim < 2 or matrix.shape[-2:] != (2, 2):
            raise ValueError('Expected (..., 2, 2) algebra matrices')
        return matrix[..., 0, :]


if __name__ == '__main__':
    pose = Affine1D.set_mat(np.array([[2., .3], [0., 1.]]))
    cm = CMTM(pose, np.array([[.2, .4], [.1, -.1]]))
    print('Adjoint:', pose.mat_adj())
    print('CMTM:', cm.mat())
