"""Canonical CMTM storage and ordinary-output boundaries."""
import math
import numpy as np
import pytest
from mathrobo import CMTM, CMVector, SE3, SO3


@pytest.mark.parametrize('element', [SO3, SE3])
@pytest.mark.parametrize('shape', [(), (2, 1)])
@pytest.mark.parametrize('order', [1, 4, 8])
def test_cm_storage_and_detached_ordinary_outputs(element, shape, order):
    raw = np.random.default_rng(42).normal(size=shape + (order-1, element.dof()))
    value = CMTM(element.eye(), raw)
    assert '_vecs' not in value.__dict__
    expected = raw / np.asarray([math.factorial(k) for k in range(order-1)])[:, None]
    np.testing.assert_allclose(value.cmvecs().cm_vecs(), expected, rtol=1e-15, atol=1e-15)
    saved = value.cmvecs().cm_vecs().copy()
    restored = value.vecs()
    np.testing.assert_allclose(restored, raw, rtol=1e-15, atol=1e-15)
    assert not np.shares_memory(restored, value.cmvecs().cm_vecs())
    restored[...] = 99
    np.testing.assert_array_equal(value.cmvecs().cm_vecs(), saved)
    supplied = CMVector.set_cmvecs(expected)
    direct = CMTM(element.eye(), supplied)
    assert direct.cmvecs() is supplied


@pytest.mark.parametrize('element', [SO3, SE3])
@pytest.mark.parametrize('order', [2, 4, 7])
def test_internal_operations_never_reconstruct_ordinary_series(monkeypatch, element, order):
    rng = np.random.default_rng(19)
    a = CMTM(element.rand(), rng.normal(scale=.1, size=(order-1, element.dof())))
    b = CMTM(element.rand(), rng.normal(scale=.1, size=(order-1, element.dof())))
    product = a.mat() @ b.mat()
    inverse = np.linalg.inv(a.mat())
    def forbidden(*args, **kwargs):
        raise AssertionError('ordinary series reconstructed inside a CMTM operation')
    monkeypatch.setattr(CMVector, 'vecs', forbidden)
    monkeypatch.setattr(CMTM, 'vecs', forbidden)
    np.testing.assert_allclose((a @ b).mat(), product, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(a.inv().mat(), inverse, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(CMTM.set_mat(element, a.mat()).mat(), a.mat(), atol=1e-13, rtol=1e-13)
