"""Boundary values and conventions, checked against independent references."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from scipy.integrate import quad_vec
from mathrobo import SO2, SE3, SE3wrench

@pytest.mark.parametrize('lib', ['numpy','jax'])
def test_so2_angle_quadrants_batch_and_jit(lib):
    angles = np.array([-3.,-2.5,-1.,0.,.7,2.5,3.])
    matrices = np.stack([SO2.angle_to_rot_mat(a) for a in angles])
    if lib == 'jax':
        actual = jax.jit(lambda m:SO2(m,LIB='jax').angle())(jnp.asarray(matrices))
    else:
        actual = SO2(matrices).angle()
    np.testing.assert_allclose(actual, angles, atol=2e-7)

@pytest.mark.parametrize('a', [0.,.7,-2.,1.])
@pytest.mark.parametrize('lib', ['numpy','jax'])
def test_full_se3_integral(a,lib):
    v = np.array([.1,.2,.3,.4,.5,.6])
    expected = quad_vec(lambda t:expm(t*SE3.hat(v)),0.,a)[0]
    if lib == 'jax':
        result = jax.jit(lambda v:SE3.exp_integ(v,a,LIB='jax'))(jnp.asarray(v))
    else:
        result = SE3.exp_integ(v,a)
    np.testing.assert_allclose(result,expected,atol=2e-6)
    np.testing.assert_allclose(result[3,3],a,atol=1e-7)

@pytest.mark.parametrize('a', [0.,.7,-2.])
@pytest.mark.parametrize('lib', ['numpy','jax'])
def test_wrench_exponential_and_integral_conventions(a,lib):
    v=np.array([.1,.2,.3,.4,.5,.6])
    generator=SE3wrench.hat_adj(v)
    expected=expm(a*generator)
    integral=quad_vec(lambda t:expm(t*generator),0.,a)[0]
    x=jnp.asarray(v) if lib=='jax' else v
    for method,ref in [('exp',expected),('exp_adj',expected),('exp_integ',integral),('exp_integ_adj',integral)]:
        f=lambda x:getattr(SE3wrench,method)(x,a,LIB=lib)
        actual=jax.jit(f)(x) if lib=='jax' else f(x)
        np.testing.assert_allclose(actual,ref,atol=2e-6)
    pose=SE3wrench.set_mat(SE3.exp(v,a))
    np.testing.assert_allclose(pose.mat_adj(),expected,atol=1e-12)

@pytest.mark.parametrize('lib', ['numpy','jax'])
@pytest.mark.parametrize('rot_batch,pos_batch', [((),(2,)),((2,),()),((2,1),(3,))])
def test_pose_broadcasts_inputs(lib,rot_batch,pos_batch):
    rot=np.broadcast_to(np.eye(3),rot_batch+(3,3))
    pos=np.broadcast_to([.1,.2,.3],pos_batch+(3,))
    if lib=='jax':
        rot,pos=jnp.asarray(rot),jnp.asarray(pos)
    p=SE3(rot,pos,LIB=lib)
    batch=np.broadcast_shapes(rot_batch,pos_batch)
    assert p.rot().shape==batch+(3,3)
    assert p.pos().shape==batch+(3,)
    np.testing.assert_allclose(p.mat()[...,:3,3],np.broadcast_to([.1,.2,.3],batch+(3,)),atol=1e-7)
    np.testing.assert_allclose(p.mat()@p.mat_inv(),np.broadcast_to(np.eye(4),batch+(4,4)),atol=1e-7)

@pytest.mark.parametrize('rot,pos', [(np.eye(2),np.zeros(3)),(np.eye(3),np.zeros(4)),(np.zeros((2,3,3)),np.zeros((3,3)))])
def test_invalid_pose_rejected_at_construction(rot,pos):
    with pytest.raises(ValueError):
        SE3(rot,pos)


def test_broadcast_pose_snapshot_and_atomic_update():
    pos=np.ones((2,3))
    p=SE3(np.eye(3),pos)
    old=p.mat()
    pos[:]=4
    np.testing.assert_array_equal(p.mat(),old)
    with pytest.raises(ValueError):
        p.update_pose(np.zeros((2,3,3)),np.zeros((3,3)))
    np.testing.assert_array_equal(p.mat(),old)
    p.update_pos(np.zeros((2,3)))
    np.testing.assert_array_equal(p.pos(),np.zeros((2,3)))


def test_jax_integral_time_derivative_is_exponential():
    v=jnp.array([.1,.2,.3,.4,.5,.6])
    for a in [0.,.7,-2.]:
        derivative=jax.jit(jax.jacfwd(lambda t:SE3.exp_integ(v,t,LIB='jax')))(a)
        np.testing.assert_allclose(derivative,SE3.exp(v,a,LIB='jax'),atol=3e-6)
        wrench_derivative=jax.jit(jax.jacfwd(lambda t:SE3wrench.exp(v,t,LIB='jax')))(a)
        expected=SE3wrench.hat_adj(v,LIB='jax')@SE3wrench.exp(v,a,LIB='jax')
        np.testing.assert_allclose(wrench_derivative,expected,atol=3e-6)


def test_jitted_pose_broadcasting():
    result=jax.jit(lambda p:SE3(jnp.eye(3),p,LIB='jax').mat())(jnp.ones((2,3)))
    assert result.shape==(2,4,4)
    np.testing.assert_array_equal(result[:,:3,3],np.ones((2,3)))
