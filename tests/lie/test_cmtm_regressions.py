"""Regressions for dtype, shared-input caches and traced transformations."""
from contextlib import contextmanager

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from mathrobo import CMTM, CMVector, SE3, SO3

@contextmanager
def x64_enabled():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


MATRICES = ['mat', 'mat_adj', 'mat_inv', 'mat_inv_adj']

@pytest.mark.parametrize('element', [SO3, SE3])
@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.complex128])
@pytest.mark.parametrize('method', MATRICES + ['tangent_mat', 'tangent_mat_cm'])
def test_series_dtype_uses_coefficients(element, dtype, method):
    raw = np.random.default_rng(4).normal(scale=.1, size=(3, element.dof())).astype(dtype if dtype != np.int64 else float)
    if dtype == np.complex128:
        raw += .02j
    coefficients = CMVector.set_cmvecs(raw)
    actual = CMTM(element.set_mat(np.eye(element.mat_size(), dtype=np.int64)), coefficients)
    reference = CMTM(element.set_mat(np.eye(element.mat_size())), coefficients)
    np.testing.assert_allclose(getattr(actual, method)(), getattr(reference, method)(), rtol=1e-6, atol=1e-8)

@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('mutation', ['translation', 'rotation', 'coefficients'])
def test_explicit_updates_invalidate_all_caches(batch, mutation):
    shape = (2,) if batch else ()
    rot = np.broadcast_to(np.eye(3), shape + (3,3)).copy()
    pos = np.zeros(shape + (3,))
    values = np.ones(shape + (4,6)) * .1
    series = CMVector.set_cmvecs(values)
    pose = SE3(rot, pos)
    actual = CMTM(pose, series)
    methods = MATRICES + ['tangent_mat', 'tangent_mat_cm']
    for method in methods:
        getattr(actual, method)()
    if mutation == 'translation':
        pos[..., 0] = 1
        pose.update_pos(pos)
        actual.update_pose(pose)
    elif mutation == 'rotation':
        rot[...] = SO3.exp(np.array([.1,.2,.3]))
        pose.update_rot(rot)
        actual.update_pose(pose)
    else:
        values[..., 0, 0] += .7
        series.update_cmvecs(values)
        actual.update_cmvecs(values)
    reference = CMTM(pose, series)
    for method in methods:
        expected = getattr(reference, method)()
        # Call apply first, to cover its independent cached fast path.
        applied = getattr(actual, 'apply_' + method)(np.ones(expected.shape[-1]))
        np.testing.assert_allclose(applied, (expected @ np.ones(expected.shape[-1])), atol=1e-12)
        np.testing.assert_allclose(getattr(actual, method)(), expected, atol=1e-12)
    np.testing.assert_allclose((actual @ reference).mat(), actual.mat() @ reference.mat(), atol=1e-12)

@pytest.mark.parametrize('element', [SO3, SE3])
@pytest.mark.parametrize('method', MATRICES)
@pytest.mark.parametrize('order', [1, 3, 7])
def test_jax_jit_apply_matches_numpy(element, method, order):
    with x64_enabled():
        values = np.random.default_rng(42).normal(scale=.1, size=(2, order-1, element.dof()))
        def action(v):
            pose = element.set_mat(jnp.eye(element.mat_size()), LIB='jax')
            cm = CMTM(pose, v, LIB='jax')
            matrix = getattr(cm, method)()
            size = matrix.shape[-1]
            return (matrix, getattr(cm, 'apply_' + method)(jnp.eye(size)),
                    getattr(cm, 'apply_' + method)(jnp.ones((2, size))),
                    getattr(cm, method)(0))
        compiled = jax.jit(action)
        for v in [values, values + .03]:
            expected = getattr(CMTM(element.eye(), v), method)()
            matrix, applied, vector, empty = compiled(jnp.asarray(v))
            np.testing.assert_allclose(matrix, expected, atol=1e-12)
            np.testing.assert_allclose(applied, expected, atol=1e-12)
            np.testing.assert_allclose(vector, expected @ np.ones(expected.shape[-1]), atol=1e-12)
            assert empty.shape == (2, 0, 0)

@pytest.mark.parametrize('method', MATRICES)
def test_jax_gradient_matches_central_difference(method):
    with x64_enabled():
        v = jnp.asarray(np.random.default_rng(8).normal(scale=.1, size=(2,6)))
        direction = jnp.ones_like(v) * .2
        def action(v):
            cm = CMTM(SE3(jnp.eye(3), jnp.zeros(3), LIB='jax'), v, LIB='jax')
            return jnp.sum(getattr(cm, method)() ** 2)
        grad = jax.jit(jax.grad(action))(v)
        step = 1e-5
        finite = (action(v+step*direction)-action(v-step*direction))/(2*step)
        np.testing.assert_allclose(jnp.sum(grad*direction), finite, atol=1e-8, rtol=1e-7)

@pytest.mark.parametrize('method', MATRICES)
def test_integer_coefficients_allow_fractional_recurrence(method):
    coefficients = np.zeros((3, 6), dtype=int)
    coefficients[0, :3] = [1, 1, 1]
    pose = SE3.set_mat(np.eye(4, dtype=int))
    actual = CMTM(pose, CMVector.set_cmvecs(coefficients))
    expected = CMTM(SE3.eye(), CMVector.set_cmvecs(coefficients.astype(float)))
    np.testing.assert_allclose(getattr(actual, method)(), getattr(expected, method)(), atol=1e-12)


def test_jax_dynamic_pose_and_vmap():
    with x64_enabled():
        v = jnp.ones((2, 6)) * .1
        def action(position):
            cm = CMTM(SE3(jnp.eye(3), position, LIB='jax'), v, LIB='jax')
            return cm.apply_mat_adj(jnp.arange(18., dtype=jnp.float64))
        positions = jnp.array([[.1,.2,.3], [-.2,.1,.4]])
        actual = jax.jit(jax.vmap(action))(positions)
        expected = [CMTM(SE3(np.eye(3), np.asarray(p)), np.asarray(v)).apply_mat_adj(np.arange(18.)) for p in positions]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        jacobian = jax.jit(jax.jacfwd(action))(positions[0])
        h = 1e-5
        finite = np.stack([(action(positions[0]+h*e)-action(positions[0]-h*e))/(2*h)
                           for e in jnp.eye(3)], axis=-1)
        np.testing.assert_allclose(jacobian, finite, atol=1e-9)
