import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from mathrobo import SE3


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_integrate_and_difference_with_broadcast_poses(dtype):
    base_coordinates = np.array([
        [[.3, -.2, .4, .5, -.1, .7]],
        [[-.5, .2, .1, -.2, .3, .4]],
    ], dtype=dtype)
    base = SE3.set_mat(SE3.exp(base_coordinates))
    axis = np.array([-.2, .3, -.9])
    axis /= np.linalg.norm(axis)
    delta = np.array([
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., .3, -.4, .8],
        [1e-8, -2e-8, 3e-8, .3, -.4, .8],
        [.00101, 0., 0., .3, -.4, .8],
        [.2, -.4, .3, .3, -.4, .8],
        [*(axis * (np.pi - 1e-4)), .3, -.4, .8],
    ], dtype=dtype)
    base_matrix = base.mat()
    updated = base.integrate(delta)
    recovered = base.difference(updated)
    reference = np.stack([expm(SE3.hat(x.astype(np.float64))) for x in delta])
    tolerance = 2e-6 if dtype == np.float32 else 2e-12
    assert updated.mat().shape == (2, len(delta), 4, 4)
    assert updated.mat().dtype == recovered.dtype == dtype
    np.testing.assert_allclose(updated.mat(), base_matrix @ reference, atol=tolerance)
    np.testing.assert_allclose(recovered, np.broadcast_to(delta, recovered.shape),
                               atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(base.mat(), base_matrix)


def test_difference_principal_branch_at_and_beyond_pi():
    base = SE3.eye()
    exact_pi = SE3(np.diag([-1., -1., 1.]), [.3, -.4, .8])
    above_pi = base.integrate(np.array([0., 0., np.pi + .01, .3, -.4, .8]))
    for pose, expected_angle in [(exact_pi, np.pi), (above_pi, np.pi - .01)]:
        delta = base.difference(pose)
        np.testing.assert_allclose(np.linalg.norm(delta[:3]), expected_angle, atol=1e-14)
        np.testing.assert_allclose(base.integrate(delta).mat(), pose.mat(), atol=1e-14)


def test_jit_pose_increment_and_derivatives_at_zero():
    base = SE3.exp(jnp.array([.3, -.2, .4, .5, -.1, .7]), LIB='jax')

    @jax.jit
    def roundtrip(delta):
        pose = SE3.set_mat(base, LIB='jax')
        return pose.difference(pose.integrate(delta))

    zero = jnp.zeros(6, dtype=base.dtype)
    pure_translation = zero.at[3:].set(jnp.array([.3, -.4, .8]))
    batch = jnp.stack((zero, pure_translation))
    np.testing.assert_allclose(roundtrip(batch), batch, atol=2e-6)
    for jacobian in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(jacobian(roundtrip)(zero), np.eye(6), atol=2e-6)


def test_invalid_pose_increment_and_difference():
    base = SE3.eye()
    with pytest.raises(ValueError, match='Increment must have shape'):
        base.integrate(0.)
    with pytest.raises(ValueError, match='matching backends'):
        base.difference(SE3.eye(LIB='jax'))
    with pytest.raises(TypeError, match='real pose'):
        SE3(np.eye(3, dtype=complex)).log()
