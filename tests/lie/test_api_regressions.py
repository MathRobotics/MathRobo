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


def _random_element(cls, rng):
    if issubclass(cls, mr.SE3):
        pose = mr.SE3.set_mat(mr.SE3.exp(rng.normal(size=6)))
        return cls(pose.rot(), pose.pos())
    return cls(mr.SO3.exp(rng.normal(size=3)))


CMTM_GROUPS = [mr.SO3, mr.SO3wrench, mr.SE3, mr.SE3wrench]


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('cls', CMTM_GROUPS)
def test_cmtm_product_and_inverse_match_matrices(cls, order):
    # Build elements directly: eye() returned the base class before it was fixed.
    rng = np.random.default_rng(order)
    left, right = (mr.CMTM(_random_element(cls, rng), rng.normal(size=(order - 1, cls.dof())))
                   for _ in range(2))
    product = left @ right

    assert type(product._mat) is cls
    np.testing.assert_allclose(product.mat(), left.mat() @ right.mat(), atol=1e-10)
    np.testing.assert_allclose(product.mat_adj(), left.mat_adj() @ right.mat_adj(), atol=1e-10)
    np.testing.assert_allclose(left.inv().mat(), np.linalg.inv(left.mat()), atol=1e-10)


@pytest.mark.parametrize('cls', CMTM_GROUPS)
def test_hat_and_vee_are_inverse(cls):
    x = np.random.default_rng(0).normal(size=cls.dof())
    np.testing.assert_allclose(cls.vee(cls.hat(x)), x)
    np.testing.assert_allclose(cls.vee_adj(cls.hat_adj(x)), x)


@pytest.mark.parametrize('cls', [mr.SO3, mr.SO3wrench, mr.SE3])
def test_hat_adj_is_lie_bracket_in_hat_coordinates(cls):
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(2, cls.dof()))
    bracket = cls.vee(cls.hat(x) @ cls.hat(y) - cls.hat(y) @ cls.hat(x))
    np.testing.assert_allclose(cls.hat_adj(x) @ y, bracket, atol=1e-12)
    np.testing.assert_allclose(cls.hat_commute_adj(y) @ x, bracket, atol=1e-12)


def test_se3wrench_hat_adj_is_coadjoint():
    x = np.random.default_rng(2).normal(size=6)
    np.testing.assert_allclose(mr.SE3wrench.hat_adj(x), -mr.SE3.hat_adj(x).T)


def test_so3wrench_exponentials_match_matrix_exponential():
    from scipy.linalg import expm
    x, a = np.array([0.3, -0.5, 0.8]), 0.7
    np.testing.assert_allclose(mr.SO3wrench.exp(x, a), expm(a * mr.SO3wrench.hat(x)), atol=1e-12)
    np.testing.assert_allclose(mr.SO3wrench.exp_adj(x, a), expm(a * mr.SO3wrench.hat_adj(x)), atol=1e-12)


def test_set_pos_quaternion_broadcasts_positions():
    positions = np.arange(9.0).reshape(3, 3)
    q = np.array([np.cos(0.2), 0.0, 0.0, np.sin(0.2)])
    poses = mr.SE3.set_pos_quaternion(positions, q)
    assert poses.mat().shape == (3, 4, 4)
    for pose, pos in zip(poses.mat(), positions):
        np.testing.assert_allclose(pose, mr.SE3.set_pos_quaternion(pos, q).mat())
    with pytest.raises(ValueError, match='Position'):
        mr.SE3.set_pos_quaternion(np.zeros(2), q)
