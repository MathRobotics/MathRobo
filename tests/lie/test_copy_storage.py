"""Copy and serialization must preserve the explicit-update ownership contract."""
import copy
import pickle
import numpy as np
import pytest
from mathrobo import SO3, SO3wrench, SE3, SE3wrench, CMVector, CMTM


def clone(value, mode):
    if mode == 'deepcopy':
        return copy.deepcopy(value)
    if mode == 'copy':
        return copy.copy(value)
    return pickle.loads(pickle.dumps(value, protocol=mode))


def protected(array):
    with pytest.raises(ValueError):
        array[...] = 0
    with pytest.raises(ValueError):
        array.setflags(write=True)


@pytest.mark.parametrize('mode', ['copy', 'deepcopy', 0, 2, pickle.HIGHEST_PROTOCOL])
@pytest.mark.parametrize('group', [SO3, SO3wrench, SE3, SE3wrench])
def test_pose_copy_keeps_readonly_storage_and_independent_updates(mode, group):
    # Use the constructor because some inherited factories return the base type.
    original = group(np.eye(3)) if issubclass(group, SO3) else group(np.eye(3), np.ones(3))
    result = clone(original, mode)
    assert type(result) is group
    protected(result.mat() if issubclass(group, SO3) else result.rot())
    before = original.mat()
    result.update_rot(SO3.exp(np.array([.1,.2,.3])))
    if isinstance(result, SE3):
        protected(result.pos())
        result.update_pos(np.zeros(3))
    np.testing.assert_array_equal(original.mat(), before)


@pytest.mark.parametrize('mode', ['copy', 'deepcopy', 0, 2, pickle.HIGHEST_PROTOCOL])
@pytest.mark.parametrize('shape', [(5,), (2,5,6)])
def test_series_copy_keeps_layout_dtype_and_update_isolation(mode, shape):
    original = CMVector(np.ones(shape, dtype=np.complex64) * (1+.25j))
    result = clone(original, mode)
    protected(result.cm_vecs())
    protected(result.cm_vec())
    assert result.cm_vecs().dtype == np.complex64
    np.testing.assert_array_equal(result.vecs(), original.vecs())
    result.update_vecs(np.zeros(shape, dtype=np.complex64))
    np.testing.assert_allclose(original.vecs(), np.ones(shape) * (1+.25j))


@pytest.mark.parametrize('mode', ['copy', 'deepcopy', 0, 2, pickle.HIGHEST_PROTOCOL])
@pytest.mark.parametrize('batch', [(), (2,)])
def test_restored_cmtm_has_consistent_storage_and_caches(mode, batch):
    pose = SE3(np.broadcast_to(np.eye(3), batch+(3,3)), np.zeros(batch+(3,)))
    values = np.full(batch+(3,6), .1)
    original = CMTM(pose, values)
    methods = ['mat', 'mat_adj', 'mat_inv', 'mat_inv_adj', 'tangent_mat', 'tangent_mat_cm']
    expected = {name:getattr(original,name)() for name in methods}
    result = clone(original, mode)
    assert all(not value for name,value in result.__dict__.items() if name.endswith('_cache_numpy'))
    protected(result.cmvecs().cm_vecs())
    for name in methods:
        np.testing.assert_allclose(getattr(result,name)(), expected[name])
    wrapper = result.cmvecs()
    wrapper.update_cmvecs(np.zeros_like(values))
    np.testing.assert_allclose(result.mat_adj(), expected['mat_adj'])
    result.update_vecs(values * 2)
    pose.update_pos(np.ones(batch+(3,)))
    result.update_pose(pose)
    reference = CMTM(pose, values * 2)
    for name in methods:
        matrix = getattr(reference,name)()
        np.testing.assert_allclose(getattr(result,'apply_'+name)(np.ones(matrix.shape[-1])),
                                   matrix @ np.ones(matrix.shape[-1]), atol=1e-12)
        np.testing.assert_allclose(getattr(original,name)(), expected[name])


@pytest.mark.parametrize('mode', ['deepcopy', pickle.HIGHEST_PROTOCOL])
def test_jax_roundtrip_keeps_backend(mode):
    import jax.numpy as jnp
    pose = SE3(jnp.eye(3), jnp.ones(3), LIB='jax')
    series = CMVector(jnp.ones((2,6)))
    source = CMTM(pose, series, LIB='jax')
    restored = clone(source, mode)
    assert isinstance(restored.cmvecs().cm_vecs(), jnp.ndarray)
    assert isinstance(restored.mat_adj(), jnp.ndarray)
    np.testing.assert_allclose(restored.mat_adj(), source.mat_adj())
