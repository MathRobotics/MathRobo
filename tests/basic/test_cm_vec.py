import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mathrobo import CMVector


@pytest.mark.parametrize("order", [0, 1, 5, 11, 14])
@pytest.mark.parametrize("batch", [(), (2, 3)])
def test_derivative_and_coefficient_conventions(order, batch):
    vecs = np.random.default_rng(0).normal(size=batch + (order, 6))
    factors = np.asarray([math.factorial(k) for k in range(order)])[:, None]
    expected_cm = vecs / factors
    normal = CMVector(vecs)
    direct = CMVector.set_cmvecs(expected_cm)

    for obj in (normal, direct):
        np.testing.assert_allclose(obj.cm_vecs(), expected_cm)
        np.testing.assert_allclose(obj.vecs(), vecs)
        np.testing.assert_allclose(obj.cm_vec(), expected_cm.reshape(batch + (order * 6,)))
        np.testing.assert_allclose(obj.vec(), vecs.reshape(batch + (order * 6,)))
    # Direct construction must retain coefficients without a lossy round trip.
    assert direct.cm_vecs() is expected_cm


def test_one_dimensional_series_layouts():
    vecs = np.arange(1.0, 6.0)
    factors = np.asarray([math.factorial(k) for k in range(len(vecs))])
    normal = CMVector(vecs)
    np.testing.assert_allclose(normal.vecs(), vecs)
    np.testing.assert_allclose(normal.cm_vecs(), vecs / factors)
    assert normal.vec().shape == normal.cm_vec().shape == (5,)

    direct = CMVector.set_cmvecs(vecs / factors)
    np.testing.assert_allclose(direct.vecs(), vecs[:, None])
    assert direct.cm_vecs().shape == (5, 1)
    assert direct.vec().shape == direct.cm_vec().shape == (5,)


@pytest.mark.parametrize("order", [5, 11, 14])
def test_jax_jit_and_gradients(order):
    x = jnp.arange(order * 6, dtype=jnp.float32).reshape(order, 6) / 10
    factors = np.asarray([math.factorial(k) for k in range(order)], dtype=np.float32)[:, None]

    @jax.jit
    def normalized(values):
        return CMVector(values).cm_vecs()

    @jax.jit
    def ordinary(values):
        return CMVector.set_cmvecs(values).vecs()

    np.testing.assert_allclose(normalized(x), np.asarray(x) / factors, rtol=2e-6)
    np.testing.assert_allclose(ordinary(normalized(x)), x, rtol=2e-6)
    gradient = jax.jit(jax.grad(lambda values: ordinary(values).sum()))(x)
    np.testing.assert_allclose(gradient, np.broadcast_to(factors, x.shape), rtol=2e-6)
    assert normalized(x).dtype == x.dtype


@pytest.mark.parametrize("order", [0, 3, 5])
@pytest.mark.parametrize("shape", [(5,), (2, 3, 5, 6)])
def test_truncate_preserves_values_and_layout(order, shape):
    ordinary = np.random.default_rng(1).normal(size=shape)
    value = CMVector(ordinary)
    shortened = value.truncate(np.int64(order))
    expected = ordinary[:order] if ordinary.ndim == 1 else ordinary[..., :order, :]
    np.testing.assert_allclose(shortened.vecs(), expected)
    assert shortened.vecs().shape == expected.shape
    assert shortened._n == order
    assert shortened._dim == value._dim
    assert shortened._len == order * value._dim
    if order:
        assert np.shares_memory(shortened.cm_vecs(), value.cm_vecs())
    assert value._n == 5


@pytest.mark.parametrize("order", [-1, 6])
def test_truncate_out_of_bounds(order):
    with pytest.raises(ValueError, match="order must be between"):
        CMVector(np.ones((5, 6))).truncate(order)


@pytest.mark.parametrize("order", [1.5, "3", None, True, np.bool_(False)])
def test_truncate_requires_integer_count(order):
    with pytest.raises(TypeError):
        CMVector(np.ones((5, 6))).truncate(order)


def test_arithmetic_broadcasts_only_batch_axes():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 1, 5, 6))
    y = rng.normal(size=(1, 3, 5, 6))
    a, b = CMVector(x), CMVector(y)
    for result, expected in [(a + b, x + y), (a - b, x - y), (-a, -x)]:
        np.testing.assert_allclose(result.vecs(), expected, atol=1e-14)
        assert not np.shares_memory(result.cm_vecs(), a.cm_vecs())
        assert not np.shares_memory(result.cm_vecs(), b.cm_vecs())
    np.testing.assert_allclose(a.vecs(), x)
    np.testing.assert_allclose(b.vecs(), y)


@pytest.mark.parametrize("shape", [(1, 6), (5, 1), (3, 6), (5, 3)])
def test_arithmetic_rejects_order_or_dimension_mismatch(shape):
    a, b = CMVector(np.ones((5, 6))), CMVector(np.ones(shape))
    with pytest.raises(ValueError, match="orders and dimensions must match"):
        a + b
    with pytest.raises(ValueError, match="orders and dimensions must match"):
        a - b


def test_arithmetic_rejects_incompatible_batch_and_operand():
    a = CMVector(np.ones((2, 5, 6)))
    b = CMVector(np.ones((3, 5, 6)))
    with pytest.raises(ValueError):
        a + b
    with pytest.raises(TypeError):
        a + 1
    with pytest.raises(TypeError):
        a - 1


def test_arithmetic_one_dimensional_layouts():
    x = np.arange(1., 6.)
    a = CMVector(x)
    b = CMVector(2 * x)
    np.testing.assert_allclose((a + b).vecs(), 3 * x)
    np.testing.assert_allclose((a - b).vecs(), -x)
    np.testing.assert_allclose((-a).vecs(), -x)
    column = CMVector(x[:, None])
    for mixed in (a + column, column + a):
        assert mixed.vecs().shape == (5, 1)
        np.testing.assert_allclose(mixed.vecs(), 2 * x[:, None])


@pytest.mark.parametrize("dtype", [np.float32, np.complex128])
def test_arithmetic_preserves_direct_coefficients(dtype):
    x = np.arange(30, dtype=np.float64).reshape(5, 6).astype(dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x += 1j * x
    a = CMVector.set_cmvecs(x)
    for result, expected in [(a + a, x + x), (a - a, x - x), (-a, -x)]:
        np.testing.assert_array_equal(result.cm_vecs(), expected)
        assert result.cm_vecs().dtype == x.dtype


def test_jax_arithmetic_and_truncation():
    x = jnp.arange(30, dtype=jnp.float32).reshape(5, 6) / 10
    y = x / 2

    @jax.jit
    def combine(a, b):
        left, right = CMVector.set_cmvecs(a), CMVector.set_cmvecs(b)
        return ((left + right) - (-right)).truncate(3).cm_vecs()

    np.testing.assert_allclose(combine(x, y), (x + 2 * y)[:3], rtol=2e-6)
    gradient = jax.jit(jax.grad(lambda a: combine(a, y).sum()))(x)
    expected = np.zeros(x.shape, dtype=np.float32)
    expected[:3] = 1
    np.testing.assert_array_equal(gradient, expected)


@pytest.mark.parametrize("input_dtype, converted_dtype", [
    (np.float32, np.float64),
    (np.float64, np.float64),
    (np.int32, np.float64),
    (np.int64, np.float64),
    (np.complex64, np.complex128),
    (np.complex128, np.complex128),
])
def test_numpy_conversion_dtype_contract(input_dtype, converted_dtype):
    values = np.arange(1, 31).reshape(5, 6).astype(input_dtype)
    if np.issubdtype(input_dtype, np.complexfloating):
        values += 1j * values
    factors = np.array([1, 1, 2, 6, 24])[:, None]
    ordinary = CMVector(values)
    direct = CMVector.set_cmvecs(values)

    assert ordinary.cm_vecs().dtype == np.dtype(converted_dtype)
    assert ordinary.cm_vec().dtype == np.dtype(converted_dtype)
    assert direct.cm_vecs().dtype == values.dtype
    assert direct.cm_vec().dtype == values.dtype
    for series, expected in [(ordinary, values), (direct, values * factors)]:
        assert series.vecs().dtype == np.dtype(converted_dtype)
        assert series.vec().dtype == np.dtype(converted_dtype)
        np.testing.assert_allclose(series.vecs(), expected)
    np.testing.assert_allclose(ordinary.cm_vecs(), values / factors)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("shape", [(5,), (2, 5, 3)])
@pytest.mark.parametrize("strided", [False, True])
def test_numpy_input_and_ordinary_output_ownership(direct, shape, strided):
    storage_shape = shape[:-1] + (shape[-1] * 2,) if strided else shape
    storage = np.arange(1., np.prod(storage_shape) + 1).reshape(storage_shape)
    source = storage[..., ::2] if strided else storage
    snapshot = source.copy()
    series = CMVector.set_cmvecs(source) if direct else CMVector(source)
    coefficients = series.cm_vecs().copy()
    expected = series.vecs().copy()

    assert np.shares_memory(series.cm_vecs(), source) == direct
    # Ordinary outputs are independent snapshots, including flattened outputs.
    for name in ("vecs", "vec"):
        output = getattr(series, name)()
        assert not np.shares_memory(output, source)
        assert not np.shares_memory(output, series.cm_vecs())
        output[...] = -999
        np.testing.assert_array_equal(source, snapshot)
        np.testing.assert_array_equal(series.cm_vecs(), coefficients)
        np.testing.assert_array_equal(series.vecs(), expected)

    source[...] = 0
    if direct:
        np.testing.assert_array_equal(series.cm_vecs(), np.zeros_like(coefficients))
    else:
        np.testing.assert_array_equal(series.cm_vecs(), coefficients)


@pytest.mark.parametrize("name", ["cm_vecs", "cm_vec"])
def test_coefficient_accessors_expose_contiguous_storage(name):
    source = np.ones((5, 6), dtype=np.float32)
    series = CMVector.set_cmvecs(source)
    accessor = getattr(series, name)()
    accessor[...] = 2
    np.testing.assert_array_equal(source, np.full_like(source, 2))
    factors = np.array([1, 1, 2, 6, 24])[:, None]
    np.testing.assert_array_equal(series.vecs(), np.broadcast_to(2 * factors, source.shape))


def test_read_only_coefficients_can_be_reconstructed():
    source = np.ones((5, 6), dtype=np.float32)
    source.setflags(write=False)
    direct = CMVector.set_cmvecs(source)
    assert not direct.cm_vecs().flags.writeable
    assert direct.vecs().flags.writeable
    with pytest.raises(ValueError):
        direct.cm_vecs()[0, 0] = 2
    ordinary = CMVector(source)
    ordinary.cm_vecs()[0, 0] = 3
    assert source[0, 0] == 1


@pytest.mark.parametrize("input_dtype, converted_dtype", [
    (jnp.float32, jnp.float32),
    (jnp.int32, jnp.float32),
    (jnp.complex64, jnp.complex64),
])
def test_jax_conversion_dtype_contract(input_dtype, converted_dtype):
    values = jnp.arange(1, 31).reshape(5, 6).astype(input_dtype)
    if jnp.issubdtype(input_dtype, jnp.complexfloating):
        values = values + 1j * values
    ordinary = CMVector(values)
    direct = CMVector.set_cmvecs(values)
    assert ordinary.cm_vecs().dtype == jnp.dtype(converted_dtype)
    assert direct.cm_vecs().dtype == values.dtype
    assert direct.vecs().dtype == jnp.dtype(converted_dtype)
    np.testing.assert_allclose(ordinary.vecs(), values, rtol=2e-6)
    changed = direct.cm_vecs().at[0, 0].set(0)
    assert changed[0, 0] == 0
    assert direct.cm_vecs()[0, 0] == values[0, 0]
