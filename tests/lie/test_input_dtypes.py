"""Input dtype policy: independent numeric references and public entry points."""
from contextlib import contextmanager

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from scipy.integrate import quad_vec
from mathrobo import SO3, SE3, SE3wrench, CMVector, CMTM


@contextmanager
def x64(enabled):
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', enabled)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.uint64, np.bool_])
@pytest.mark.parametrize('group', [SO3, SE3, SE3wrench])
def test_integer_pose_storage_and_updates(dtype, group):
    pose = group.set_mat(np.eye(group.mat_size(), dtype=dtype))
    assert pose.mat().dtype == np.float64
    pose.update_rot(np.eye(3, dtype=dtype))
    if group is not SO3:
        pose.update_pos(np.array([.25, .5, .75]))
        np.testing.assert_array_equal(pose.mat()[:3, 3], [.25, .5, .75])
        pose.update_pose(np.eye(3, dtype=dtype), np.array([1, 2, 3], dtype=dtype))
        assert pose.pos().dtype == pose.rot().dtype == np.float64
    np.testing.assert_allclose(pose.mat() @ pose.mat_inv(), np.eye(group.mat_size()))
    np.testing.assert_allclose(pose.mat_adj() @ pose.mat_inv_adj(), np.eye(group.mat_adj_size()))


@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.complex64, np.complex128])
def test_precision_preserved_at_storage_boundaries(dtype):
    rot = np.eye(3, dtype=dtype)
    pos = np.ones(3, dtype=dtype)
    values = np.ones((3, 6), dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        pos += .25j
        values += .125j
    pose = SE3(rot, pos)
    pose.update_pose(rot, pos)
    assert pose.rot().dtype == pose.pos().dtype == np.dtype(dtype)
    np.testing.assert_array_equal(pose.mat()[:3, 3], pos)
    for series in (CMVector(values), CMVector.set_cmvecs(values)):
        series.update_vecs(values)
        assert series.cm_vecs().dtype == series.vecs().dtype == np.dtype(dtype)
        np.testing.assert_allclose(series.vecs(), values, rtol=2e-6)
        series.update_cmvecs(values)
        cm = CMTM(pose, series)
        cm.update_cmvecs(values)
        assert cm.mat_adj().dtype == np.dtype(dtype)


@pytest.mark.parametrize('group', [SE3, SE3wrench])
def test_mixed_complex_position_is_not_discarded(group):
    pose = group(np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    pose.update_pos(np.array([.25+.5j, .5, .75], dtype=np.complex64))
    assert pose.rot().dtype == pose.pos().dtype == np.complex64
    np.testing.assert_array_equal(pose.mat()[:3, 3], [.25+.5j, .5, .75])
    reference = group(np.eye(3, dtype=np.complex128), pose.pos().astype(np.complex128))
    for method in ('mat', 'mat_inv', 'mat_adj', 'mat_inv_adj'):
        np.testing.assert_allclose(getattr(pose, method)(), getattr(reference, method)(), atol=1e-6)


@pytest.mark.parametrize('group,values', [(SO3, [1,0,1]), (SE3, [1,0,1,1,2,3])])
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.uint64, np.bool_])
@pytest.mark.parametrize('adjoint', [False, True])
def test_integer_exponentials_against_scipy(group, values, dtype, adjoint):
    v = np.array(values, dtype=dtype)
    hat = group.hat_adj if adjoint else group.hat
    exp = group.exp_adj if adjoint else group.exp
    # SciPy supplies the independent matrix-exponential reference.
    generator = hat(v.astype(float))
    result = exp(v, .7)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expm(.7 * generator), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(result[:3,:3]), 1., atol=1e-12)


@pytest.mark.parametrize('group', [SO3, SE3])
@pytest.mark.parametrize('adjoint', [False, True])
def test_integer_integrals_against_quadrature(group, adjoint):
    v = np.arange(1, group.dof()+1, dtype=int)
    hat = group.hat_adj if adjoint else group.hat
    integ = group.exp_integ_adj if adjoint else group.exp_integ
    expected = quad_vec(lambda t: expm(t * hat(v.astype(float))), 0., 1.)[0]
    np.testing.assert_allclose(integ(v, 1.), expected, atol=1e-11)


@pytest.mark.parametrize('group', [SO3, SE3])
def test_integer_vee_keeps_fractional_average(group):
    matrix = np.zeros((group.mat_size(), group.mat_size()), dtype=int)
    matrix[1, 0] = 1
    result = group.vee(matrix)
    assert result.dtype == np.float64
    assert result[2] == .5


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.uint64, np.bool_])
def test_integer_cm_updates_keep_floating_storage(dtype):
    values = np.ones((3, 6), dtype=dtype)
    for series in (CMVector(values), CMVector.set_cmvecs(values)):
        assert series.cm_vecs().dtype == np.float64
        series.update_vecs(values)
        assert series.cm_vecs().dtype == np.float64
        np.testing.assert_allclose(series.vecs(), values)
        series.update_cmvecs(values)
        cm = CMTM(SE3.eye(), series)
        cm.mat_adj()
        cm.update_vecs(values)
        assert cm.cmvecs().cm_vecs().dtype == np.float64
        np.testing.assert_allclose(cm.vecs(), values)
        cm.update_cmvecs(values)
        np.testing.assert_array_equal(cm.cmvecs().cm_vecs(), values)


@pytest.mark.parametrize('enabled', [False, True])
def test_jax_integer_inputs_and_jitted_updates(enabled):
    with x64(enabled):
        expected_dtype = jnp.float64 if enabled else jnp.float32
        @jax.jit
        def action(v):
            pose = SE3(jnp.eye(3, dtype=jnp.int32), jnp.zeros(3, dtype=jnp.int32), LIB='jax')
            pose.update_pos(jnp.array([.25,.5,.75]))
            cm = CMTM(pose, CMVector.set_cmvecs(v), LIB='jax')
            cm.update_cmvecs(v)
            return pose.mat(), SE3.exp(v[0], .7, LIB='jax'), cm.mat_adj()
        v = jnp.ones((2,6), dtype=jnp.int32)
        pose, exp, cm = action(v)
        for value in (pose, exp, cm):
            assert value.dtype == expected_dtype
            assert np.isfinite(value).all()
        np.testing.assert_allclose(pose[:3, 3], [.25,.5,.75])
        np.testing.assert_allclose(exp, expm(.7 * SE3.hat(np.ones(6))), atol=2e-6)


@pytest.mark.parametrize('group', [SO3, SE3])
@pytest.mark.parametrize('method', ['hat', 'hat_adj', 'exp', 'exp_integ', 'exp_integ_adj'])
@pytest.mark.parametrize('dtype', [np.float32, np.complex64])
def test_direct_operations_preserve_precision(group, method, dtype):
    value = np.ones(group.dof(), dtype=dtype) * .1
    if np.issubdtype(dtype, np.complexfloating):
        value += .01j
    operation = getattr(group, method)
    result = operation(value, .7) if method.startswith('exp') else operation(value)
    assert result.dtype == np.dtype(dtype)


def test_integer_cmtm_matrix_recovery():
    source = CMTM(SE3.eye(), np.ones((1,6))).mat().astype(int)
    recovered = CMTM.set_mat(SE3, source)
    assert recovered.cmvecs().cm_vecs().dtype == np.float64
    np.testing.assert_allclose(recovered.mat(), source)
