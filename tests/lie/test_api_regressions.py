import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import mathrobo as mr

ORDERS = ['ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY']


@pytest.mark.parametrize('LIB', ['numpy', 'jax'])
@pytest.mark.parametrize('order', ORDERS)
def test_set_euler_matches_intrinsic_axis_product(order, LIB):
    angles = np.array([0.3, -0.4, 0.7])
    by_axis = dict(zip('XYZ', angles))
    expected = Rotation.from_euler(order, [by_axis[axis] for axis in order]).as_matrix()

    rot = np.asarray(mr.SO3.set_euler(angles, order, LIB).mat())

    np.testing.assert_allclose(rot, expected, atol=1e-6)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-6)


def test_set_euler_supports_batches_and_rejects_unknown_order():
    angles = np.random.default_rng(0).normal(size=(4, 3))
    batch = mr.SO3.set_euler(angles, 'YXZ').mat()
    for single, angle in zip(batch, angles):
        np.testing.assert_allclose(single, mr.SO3.set_euler(angle, 'YXZ').mat())
    with pytest.raises(ValueError, match='Unsupported order'):
        mr.SO3.set_euler(angles[0], 'XYX')


def _random_rotations(count):
    return Rotation.random(count, random_state=1).as_matrix()


def test_quaternion_round_trip_for_batches_and_all_branches():
    rotations = np.concatenate([
        _random_rotations(32),
        # Trace <= 0 with each diagonal entry as the maximum.
        np.stack([np.diag([1., -1., -1.]), np.diag([-1., 1., -1.]), np.diag([-1., -1., 1.])]),
    ])
    quaternions = mr.SO3(rotations).quaternion()
    assert quaternions.shape == (len(rotations), 4)
    for rot, q in zip(rotations, quaternions):
        np.testing.assert_allclose(mr.SO3.quaternion_to_mat(q), rot, atol=1e-12)
        np.testing.assert_allclose(q, mr.SO3(rot).quaternion())


def test_quaternion_is_jittable_with_jax():
    rot = _random_rotations(1)[0]
    q = jax.jit(lambda r: mr.SO3(r, 'jax').quaternion())(jnp.asarray(rot))
    assert isinstance(q, jnp.ndarray)
    np.testing.assert_allclose(np.asarray(q), mr.SO3(rot).quaternion(), atol=1e-6)


@pytest.mark.parametrize('cls', [mr.SO3, mr.SO3wrench, mr.SE3, mr.SE3wrench, mr.SE3inertia])
def test_constructors_and_products_preserve_subclass(cls):
    a = cls.eye()
    b = cls.set_mat(a.mat())
    assert type(a) is cls
    assert type(b) is cls
    assert type(a @ b) is cls
    assert type(a.inv()) is cls


def test_se3wrench_product_uses_wrench_adjoint():
    left, right = mr.SE3wrench.rand(), mr.SE3wrench.rand()
    np.testing.assert_allclose((left @ right).mat_adj(), left.mat_adj() @ right.mat_adj(), atol=1e-12)


def test_invalid_frame_is_rejected():
    pose = mr.SE3.eye()
    with pytest.raises(ValueError, match='frame'):
        pose.mat_var_x_arb_vec(np.ones(6), np.ones(6), frame='body')
    with pytest.raises(ValueError, match='frame'):
        pose.mat_var_x_arb_vec_jacob(np.ones(6), frame='body')
    with pytest.raises(ValueError, match='frame'):
        mr.SE3.sub_tan_vec(pose, pose, 'body')
    with pytest.raises(ValueError, match='frame'):
        mr.SO3.sub_tan_vec(mr.SO3.eye(), mr.SO3.eye(), 'body')
    cmtm = mr.CMTM.rand(mr.SE3, 3)
    with pytest.raises(ValueError, match='frame'):
        cmtm.mat_var_x_arb_vec_jacob(mr.CMVector(np.ones((3, 6))), frame='body')


@pytest.mark.parametrize('order', [1, 2, 3, 4])
def test_cmtm_product_is_jittable_with_jax(order):
    rng = np.random.default_rng(order)
    poses = [mr.SE3.set_mat(mr.SE3.exp(rng.normal(size=6))) for _ in range(2)]
    vecs = [rng.normal(size=(order - 1, 6)) for _ in range(2)]
    expected = (mr.CMTM(poses[0], vecs[0]) @ mr.CMTM(poses[1], vecs[1])).mat()

    def product(left_vecs, right_vecs):
        left = mr.CMTM(mr.SE3.set_mat(jnp.asarray(poses[0].mat()), 'jax'), left_vecs, LIB='jax')
        right = mr.CMTM(mr.SE3.set_mat(jnp.asarray(poses[1].mat()), 'jax'), right_vecs, LIB='jax')
        return (left @ right).mat()

    result = jax.jit(product)(jnp.asarray(vecs[0]), jnp.asarray(vecs[1]))
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4, atol=1e-4)
