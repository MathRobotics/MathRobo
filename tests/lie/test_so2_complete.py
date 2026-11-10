import copy
import pickle
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from scipy.integrate import quad_vec
from mathrobo import SO2, CMTM

@pytest.mark.parametrize('lib', ['numpy','jax'])
@pytest.mark.parametrize('a', [0.,.7,-2.])
def test_exponential_integrals_and_adjoint(lib,a):
    xp=np if lib=='numpy' else jnp
    values=xp.asarray([[0.],[1e-9],[.3],[-2.]])
    for v in values:
        generator=np.array([[0.,-float(v[0])],[float(v[0]),0.]])
        for method,expected in [('exp',expm(a*generator)),
                ('exp_integ',quad_vec(lambda t:expm(t*generator),0.,a)[0]),
                ('exp_integ2nd',quad_vec(lambda t:(a-t)*expm(t*generator),0.,a)[0])]:
            fn=lambda x:getattr(SO2,method)(x,a,LIB=lib)
            actual=jax.jit(fn)(v) if lib=='jax' else fn(v)
            np.testing.assert_allclose(actual,expected,atol=2e-6)
    assert SO2.hat(values,lib).shape==(4,2,2)
    np.testing.assert_allclose(SO2.vee(SO2.hat(values,lib),lib),values)
    np.testing.assert_array_equal(SO2.hat_adj(values,lib),np.zeros((4,1,1)))
    np.testing.assert_array_equal(SO2.exp_adj(values,a,lib),np.ones((4,1,1)))
    np.testing.assert_allclose(SO2.exp_integ_adj(values,a,lib),a*np.ones((4,1,1)))

@pytest.mark.parametrize('lib', ['numpy','jax'])
def test_batch_group_and_backend(lib):
    p=SO2.set_angle(np.array([.1,.2,.3]),LIB=lib)
    assert p.mat().shape==(3,2,2)
    assert p.inv().lib==lib
    np.testing.assert_allclose((p@p.inv()).mat(),np.broadcast_to(np.eye(2),(3,2,2)),atol=1e-7)
    assert (p@p).lib==lib
    np.testing.assert_allclose((p@p).angle(),[.2,.4,.6],atol=1e-7)
    assert p.mat_inv().shape==(3,2,2)
    np.testing.assert_array_equal(p.mat_adj(),np.ones((3,1,1)))
    np.testing.assert_allclose(p@np.array([1.,0.]),np.asarray(p.mat())[...,0],atol=1e-7)
    np.testing.assert_allclose(SO2.hat([.2],lib)@np.array([2.,3.]),SO2.hat_commute(np.array([2.,3.]),lib)@np.array([.2]),atol=1e-7)

@pytest.mark.parametrize('clone',[copy.copy,copy.deepcopy,lambda x:pickle.loads(pickle.dumps(x))])
def test_storage_update_copy(clone):
    source=np.eye(2,dtype=int)
    p=SO2(source); source[:]=0
    assert p.mat().dtype==np.float64
    q=clone(p)
    with pytest.raises(ValueError):q.mat()[0,0]=0
    with pytest.raises(ValueError):q.mat().setflags(write=True)
    q.update_rot(SO2.angle_to_rot_mat(.5))
    np.testing.assert_array_equal(p.mat(),np.eye(2))
    with pytest.raises(ValueError):q.update_rot(np.eye(3))
    np.testing.assert_allclose(q.angle(),.5)

@pytest.mark.parametrize('batch',[(),(3,)])
def test_cmtm_matrix_inverse_composition_recovery(batch):
    pose=SO2.set_angle(np.full(batch,.3))
    velocity=np.broadcast_to([[.2],[.1],[.05]],batch+(3,1))
    c=CMTM(pose,velocity)
    matrix=c.mat()
    assert matrix.shape==batch+(8,8)
    assert c.adj_size()==4
    np.testing.assert_allclose(c.mat_adj(),np.broadcast_to(np.eye(4),batch+(4,4)))
    np.testing.assert_allclose(c.mat_inv_adj(),c.mat_adj())
    np.testing.assert_allclose(matrix@c.mat_inv(),np.broadcast_to(np.eye(8),batch+(8,8)),atol=1e-12)
    np.testing.assert_allclose(c.inv().mat(),c.mat_inv(),atol=1e-12)
    np.testing.assert_allclose((c@c).mat(),matrix@matrix,atol=1e-12)
    recovered=CMTM.set_mat(SO2,matrix)
    np.testing.assert_allclose(recovered.vecs(),velocity,atol=1e-12)
    c.update_pose(SO2.set_angle(np.full(batch,.6)))
    np.testing.assert_allclose(c.mat(),CMTM(SO2.set_angle(np.full(batch,.6)),velocity).mat())


def test_jit_derivatives_and_cmtm():
    for method in ['exp','exp_integ','exp_integ2nd']:
        f=lambda v:getattr(SO2,method)(v,.7,LIB='jax')
        actual=jax.jit(jax.jacrev(f))(jnp.zeros(1))[...,0]
        factor={'exp':.7,'exp_integ':.7**2/2,'exp_integ2nd':.7**3/6}[method]
        np.testing.assert_allclose(actual,factor*np.array([[0.,-1.],[1.,0.]]),atol=1e-7)
    def action(v):
        cm=CMTM(SO2.set_angle(.2,LIB='jax'),v,LIB='jax')
        return cm.mat(),cm.mat_inv(),cm.apply_mat_adj(jnp.ones(v.shape[0]+1))
    v=jnp.array([[.2],[.1]])
    m,inv,adj=jax.jit(action)(v)
    np.testing.assert_allclose(m@inv,np.eye(6),atol=1e-7)
    np.testing.assert_array_equal(adj,np.ones(3))
    assert np.isfinite(jax.jit(jax.jacrev(lambda v:action(v)[0]))(v)).all()


def test_nonrecoverable_adjoint_and_legacy_alias():
    with pytest.raises(ValueError):SO2.vee_adj(np.zeros((1,1)))
    with pytest.raises(ValueError):SO2.set_mat_adj(np.eye(1))
    with pytest.warns(DeprecationWarning):p=SO2.set_quaternion(.4)
    np.testing.assert_allclose(p.angle(),.4)
    assert SO2.dof()==1 and SO2.mat_size()==2 and SO2.mat_adj_size()==1
    with pytest.raises(ValueError):SO2.hat(np.ones(2))
    with pytest.raises(ValueError):SO2(np.eye(3))
