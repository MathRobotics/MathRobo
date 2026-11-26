import jax
import numpy as np
import pytest

import mathrobo as mr

RAND_CLASSES = [mr.SO2, mr.SO3, mr.SO3wrench, mr.SE3, mr.SE3wrench]


@pytest.mark.parametrize('LIB', ['numpy', 'jax'])
@pytest.mark.parametrize('cls', RAND_CLASSES)
def test_rand_varies_and_is_reproducible(cls, LIB):
    first, second = cls.rand(LIB), cls.rand(LIB)
    assert type(first) is cls
    assert not np.allclose(first.mat(), second.mat())

    np.testing.assert_array_equal(cls.rand(LIB, key=7).mat(), cls.rand(LIB, key=7).mat())
    np.random.seed(11)
    seeded = cls.rand(LIB).mat()
    np.random.seed(11)
    np.testing.assert_array_equal(seeded, cls.rand(LIB).mat())


def test_rand_accepts_backend_native_keys():
    np.testing.assert_array_equal(mr.SE3.rand('jax', key=jax.random.PRNGKey(3)).mat(),
                                  mr.SE3.rand('jax', key=3).mat())
    np.testing.assert_array_equal(mr.SE3.rand(key=np.random.default_rng(3)).mat(),
                                  mr.SE3.rand(key=3).mat())
    with pytest.raises(TypeError):
        mr.SO3.rand(key='seed')
    with pytest.raises(ValueError):
        mr.SO3.rand('torch')


def test_cmtm_rand_is_reproducible_with_key():
    a, b = mr.CMTM.rand(mr.SE3, 4, key=5), mr.CMTM.rand(mr.SE3, 4, key=5)
    np.testing.assert_array_equal(a.mat(), b.mat())
    assert not np.allclose(a.mat(), mr.CMTM.rand(mr.SE3, 4, key=6).mat())


def test_so3_rand_is_haar_distributed():
    rotations = np.stack([mr.SO3.rand(key=np.random.default_rng(i)).mat() for i in range(4000)])
    np.testing.assert_allclose(rotations @ rotations.transpose(0, 2, 1),
                               np.broadcast_to(np.eye(3), rotations.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-12)
    # Haar measure: E[R] = 0 and the angle has density (1 - cos t) / pi on [0, pi].
    np.testing.assert_allclose(rotations.mean(axis=0), np.zeros((3, 3)), atol=0.05)
    angle = np.arccos(np.clip((np.trace(rotations, axis1=1, axis2=2) - 1) / 2, -1, 1))
    assert abs(angle.mean() - (np.pi / 2 + 2 / np.pi)) < 0.03
