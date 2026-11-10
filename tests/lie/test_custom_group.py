import copy
import pickle
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from mathrobo import MatrixLieGroup, CMTM
from examples.custom_lie_group import Affine1D


def test_required_methods_and_unsupported_recovery():
    with pytest.raises(TypeError):MatrixLieGroup(np.eye(2))
    class Missing(MatrixLieGroup):
        @staticmethod
        def dof():return 1
    with pytest.raises(TypeError):Missing(np.eye(2))
    with pytest.raises(NotImplementedError):Affine1D.vee_adj(np.eye(2))
    with pytest.raises(NotImplementedError):Affine1D.set_mat_adj(np.eye(2))

@pytest.mark.parametrize('lib',['numpy','jax'])
def test_generic_group_and_algebra(lib):
    p=Affine1D.set_mat(np.array([[2.,.3],[0.,1.]]),LIB=lib)
    np.testing.assert_allclose(p.mat_adj(),[[1.,0.],[-.3,2.]],atol=1e-7)
    np.testing.assert_allclose(p.mat_inv_adj()@p.mat_adj(),np.eye(2),atol=1e-7)
    np.testing.assert_allclose((p@p.inv()).mat(),np.eye(2),atol=1e-7)
    v=np.array([.2,.4]);w=np.array([.3,-.1])
    np.testing.assert_allclose(Affine1D.hat_adj(v,lib),[[0.,0.],[-.4,.2]],atol=1e-7)
    np.testing.assert_allclose(Affine1D.hat_commute(w,lib)@v,Affine1D.hat(v,lib)@w,atol=1e-7)
    np.testing.assert_allclose(Affine1D.hat_commute_adj(w,lib)@v,Affine1D.hat_adj(v,lib)@w,atol=1e-7)
    expected=np.array([[np.exp(.2),.4*np.expm1(.2)/.2],[0.,1.]])
    np.testing.assert_allclose(Affine1D.exp(v,LIB=lib),expected,atol=2e-7)
    # Nilpotent translation has exactly integrable polynomial exponential.
    trans=np.array([0.,.4])
    np.testing.assert_allclose(Affine1D.exp_integ(trans,.7,lib),[[.7,.4*.7**2/2],[0.,.7]],atol=2e-7)
    np.testing.assert_allclose(Affine1D.exp_integ2nd(trans,.7,lib),[[.7**2/2,.4*.7**3/6],[0.,.7**2/2]],atol=2e-7)

@pytest.mark.parametrize('batch',[(),(2,)])
def test_custom_cmtm(batch):
    pose=Affine1D.set_mat(np.broadcast_to([[2.,.3],[0.,1.]],batch+(2,2)))
    values=np.broadcast_to([[.2,.4],[.1,-.2],[.05,.1]],batch+(3,2))
    cm=CMTM(pose,values)
    m=cm.mat()
    np.testing.assert_allclose(m@cm.mat_inv(),np.broadcast_to(np.eye(8),batch+(8,8)),atol=1e-12)
    np.testing.assert_allclose((cm@cm).mat(),m@m,atol=1e-12)
    np.testing.assert_allclose(cm.inv().mat(),cm.mat_inv(),atol=1e-12)
    recovered=CMTM.set_mat(Affine1D,m)
    assert isinstance(recovered._mat,Affine1D)
    np.testing.assert_allclose(recovered.vecs(),values,atol=1e-12)
    np.testing.assert_allclose(cm.mat_adj()@cm.mat_inv_adj(),np.broadcast_to(np.eye(8),batch+(8,8)),atol=1e-12)
    np.testing.assert_allclose(cm.apply_mat_adj(np.ones(8)),cm.mat_adj()@np.ones(8),atol=1e-12)
    cm.update_pose(Affine1D.set_mat(np.broadcast_to(np.eye(2),batch+(2,2))))
    np.testing.assert_allclose(cm.elem_mat(),np.broadcast_to(np.eye(2),batch+(2,2)))


def test_storage_and_jit():
    p=Affine1D.eye();saved=p.mat()
    for clone in [copy.deepcopy(p),pickle.loads(pickle.dumps(p))]:
        with pytest.raises(ValueError):clone.mat().setflags(write=True)
        clone.update_mat(np.array([[2.,1.],[0.,1.]]))
        np.testing.assert_array_equal(p.mat(),saved)
    def action(v):
        pose=Affine1D.eye(LIB='jax')
        cm=CMTM(pose,v,LIB='jax')
        return cm.mat(),cm.mat_adj(),Affine1D.exp(v[0],LIB='jax')
    v=jnp.array([[.2,.4],[.1,.3]])
    actual=jax.jit(action)(v)
    expected=CMTM(Affine1D.eye(),np.asarray(v))
    np.testing.assert_allclose(actual[0],expected.mat(),atol=1e-7)
    np.testing.assert_allclose(actual[1],expected.mat_adj(),atol=1e-7)
    assert np.isfinite(jax.jit(jax.jacrev(lambda v:action(v)[0]))(v)).all()
    matrices=jax.jit(lambda x:Affine1D.exp(x,LIB='jax'))(v)
    assert matrices.shape==(2,2,2)
