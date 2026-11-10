"""Zero/near-zero derivatives compared with independent matrix exponential calculus."""
from contextlib import contextmanager
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from scipy.linalg import expm_frechet
from scipy.integrate import quad_vec
from mathrobo import SO3, SE3


@contextmanager
def x64():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
        yield
    finally:
        jax.config.update('jax_enable_x64', previous)


CASES = [(SO3, 'exp'), (SO3, 'exp_integ'), (SO3, 'exp_integ2nd'),
         (SE3, 'exp'), (SE3, 'exp_adj'), (SE3, 'exp_integ'), (SE3, 'exp_integ_adj')]

@pytest.mark.parametrize('group,method', CASES)
@pytest.mark.parametrize('scale', [0., 1e-9, .2])
def test_forward_reverse_derivatives_match_frechet(group, method, scale):
    with x64():
        v = np.arange(1, group.dof()+1) * scale
        if group is SE3:
            v[3:] = [.3,-.2,.4]  # Pure translation is also a zero-angle case.
        direction = np.arange(1, group.dof()+1) * .1
        a = .7
        hat = group.hat_adj if method.endswith('_adj') else group.hat
        generator, tangent = hat(v), hat(direction)
        def derivative(t):
            return expm_frechet(t*generator, t*tangent, compute_expm=False)
        if method == 'exp_integ2nd':
            expected = quad_vec(lambda t:(a-t)*derivative(t), 0., a)[0]
        elif 'integ' in method:
            expected = quad_vec(derivative, 0., a)[0]
        else:
            expected = derivative(a)
        f = lambda x:getattr(group,method)(x,a,LIB='jax')
        forward = jax.jit(jax.jacfwd(f))(jnp.asarray(v)) @ direction
        reverse = jax.jit(jax.jacrev(f))(jnp.asarray(v)) @ direction
        np.testing.assert_allclose(forward, expected, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(reverse, expected, atol=2e-10, rtol=2e-9)


@pytest.mark.parametrize('group,method', CASES)
def test_high_derivatives_at_zero(group, method):
    with x64():
        direction = jnp.arange(1., group.dof()+1) * .1
        a = .7
        hat = group.hat_adj if method.endswith('_adj') else group.hat
        generator = np.asarray(hat(direction, LIB='jax'))
        f = lambda t:getattr(group,method)(t*direction,a,LIB='jax')
        for order in range(1,5):
            f = jax.jacfwd(f)
            actual = jax.jit(f)(0.)
            scale = a**order
            if method == 'exp_integ2nd':
                scale = a**(order+2)/((order+1)*(order+2))
            elif 'integ' in method:
                scale = a**(order+1)/(order+1)
            expected = scale * np.linalg.matrix_power(generator, order)
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-10)


def test_reverse_second_derivative_and_vmap_at_zero():
    with x64():
        direction = jnp.array([.2,-.1,.3])
        scalar = lambda t:jnp.sum(SO3.exp(t*direction, .7, LIB='jax'))
        second = jax.jit(jax.grad(jax.grad(scalar)))(0.)
        g = np.asarray(SO3.hat(direction, LIB='jax'))
        np.testing.assert_allclose(second, np.sum(.7**2 * g@g), atol=1e-12)
        batch = jnp.array([[0.,0.,0.],[1e-10,0.,0.],[.1,.2,.3]])
        actual = jax.jit(jax.vmap(jax.jacrev(lambda x:SO3.exp(x,LIB='jax'))))(batch)
        assert np.isfinite(actual).all()


def test_float32_reverse_derivative_and_zero_time():
    previous = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', False)
    try:
        direction = jnp.array([.2,-.1,.3], dtype=jnp.float32)
        derivative = jax.jit(jax.jacrev(lambda x:SO3.exp(x, LIB='jax')))(jnp.zeros(3))
        np.testing.assert_allclose(derivative @ direction, SO3.hat(direction, LIB='jax'), atol=1e-6)
        time_derivative = jax.jit(jax.jacrev(lambda a:SO3.exp(direction, a, LIB='jax')))(0.)
        np.testing.assert_allclose(time_derivative, SO3.hat(direction, LIB='jax'), atol=1e-6)
    finally:
        jax.config.update('jax_enable_x64', previous)
