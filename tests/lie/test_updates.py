"""Ownership and explicit update contracts."""
import warnings

import numpy as np
import pytest
from mathrobo import SO3, SE3, CMVector, CMTM

@pytest.mark.parametrize('dtype', [np.float32, np.int64, np.complex128])
def test_pose_snapshot_and_atomic_updates(dtype):
    rot = np.eye(3, dtype=dtype)
    pos = np.zeros(3, dtype=dtype)
    pose = SE3(rot, pos)
    rotation = SO3(rot)
    rot[:] = 0
    pos[:] = 9
    np.testing.assert_array_equal(pose.rot(), np.eye(3))
    np.testing.assert_array_equal(pose.pos(), np.zeros(3))
    for a in (pose.rot(), pose.pos(), rotation.mat()):
        with pytest.raises(ValueError):
            a[...] = 7
        with pytest.raises(ValueError):
            a.setflags(write=True)
    old = pose.mat()
    with pytest.raises(ValueError):
        pose.update_pose(np.zeros((3,3)), np.zeros(4))
    np.testing.assert_array_equal(pose.mat(), old)
    pose.update_pose(np.eye(3, dtype=dtype), pos)
    pos[:] = 0
    np.testing.assert_array_equal(pose.pos(), np.full(3, 9))
    rotation.update_rot(np.eye(3) * 2)
    np.testing.assert_array_equal(rotation.mat(), np.eye(3) * 2)

@pytest.mark.parametrize('order', [1, 2, 5])
def test_cmtm_isolated_updates_and_output_safety(order):
    pose = SE3.eye()
    series = CMVector(np.ones((order-1, 6)))
    cm = CMTM(pose, series)
    original = cm.mat_adj()
    pose.update_pos(np.ones(3))
    series.update_cmvecs(np.zeros((order-1, 6)))
    wrapper = cm.cmvecs()
    wrapper.update_cmvecs(np.zeros((order-1, 6)))
    np.testing.assert_array_equal(cm.mat_adj(), original)
    cm.mat_adj()[...] = 0
    np.testing.assert_array_equal(cm.mat_adj(), original)
    cm.update_pose(pose)
    cm.update_vecs(np.zeros((order-1, 6)))
    np.testing.assert_allclose(cm.mat_adj(), CMTM(pose, series).mat_adj())
    current = cm.mat_adj()
    with pytest.raises(ValueError):
        cm.update_cmvecs(np.zeros((order, 6)))
    with pytest.raises(TypeError):
        cm.update_pose(SO3.eye())
    np.testing.assert_array_equal(cm.mat_adj(), current)

@pytest.mark.parametrize('shape', [(5,), (2,5,6)])
def test_truncate_and_accessors_remain_snapshots(shape):
    series = CMVector(np.ones(shape))
    prefix = series.truncate(3)
    saved = prefix.vecs().copy()
    old = series.cm_vecs()
    series.update_vecs(np.ones(shape) * 2)
    np.testing.assert_allclose(prefix.vecs(), saved)
    assert not np.shares_memory(old, series.cm_vecs())
    prefix.update_vecs(saved * 3)
    np.testing.assert_allclose(series.vecs(), np.ones(shape) * 2)


def test_getter_shape_metadata_does_not_mutate_owner():
    pose = SE3.eye()
    rotation = SO3.eye()
    series = CMVector(np.ones((2, 6)))
    for getter, expected in [(pose.rot, (3,3)), (pose.pos, (3,)),
                             (rotation.mat, (3,3)), (series.cm_vecs, (2,6))]:
        exposed = getter()
        # Deliberately exercise metadata mutation (deprecated in NumPy 2.5).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array",
                                    category=DeprecationWarning)
            exposed.shape = (exposed.size, 1)
        assert getter().shape == expected
