import importlib

import pytest

import mathrobo as mr

SUBPACKAGES = {'basic', 'calculus', 'lie', 'transformation'}


def test_top_level_namespace_is_exactly_the_public_api():
    public = {name for name in dir(mr) if not name.startswith('_')}
    assert public == set(mr.__all__) | SUBPACKAGES
    assert len(mr.__all__) == len(set(mr.__all__))


@pytest.mark.parametrize('leaked', ['np', 'jnp', 'jax', 'BSpline', 'snapshot', 'taylor_matmul',
                                    'check_frame', 'SnapshotState', 'Union'])
def test_internal_names_are_not_exported(leaked):
    assert not hasattr(mr, leaked)


@pytest.mark.parametrize('module', [
    'mathrobo.basic.basic', 'mathrobo.basic.cm_vec', 'mathrobo.basic.factorial_vec',
    'mathrobo.calculus.integral', 'mathrobo.calculus.numerical_grad', 'mathrobo.calculus.bspline',
    'mathrobo.transformation.rotation', 'mathrobo.lie.lie_abst', 'mathrobo.lie.so2',
    'mathrobo.lie.so3', 'mathrobo.lie.se3', 'mathrobo.lie.cmtm_abst', 'mathrobo.lie.matrix_lie_group',
])
def test_module_all_lists_names_defined_there(module):
    mod = importlib.import_module(module)
    for name in mod.__all__:
        assert getattr(mod, name).__module__ == mod.__name__
        assert getattr(mr, name) is getattr(mod, name)


def test_cmvec_module_alias_is_deprecated():
    with pytest.warns(DeprecationWarning, match='CMVector'):
        assert mr.cmvec.CMVector is mr.CMVector
