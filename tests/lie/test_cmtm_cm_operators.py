import math

import numpy as np
import pytest

import mathrobo as mr


@pytest.mark.parametrize("group", [mr.SO3, mr.SE3])
@pytest.mark.parametrize("batch", [(), (2, 3)])
@pytest.mark.parametrize("order", [0, 1, 5])
def test_cm_hat_and_vee_round_trip(group, batch, order):
    ordinary = np.random.default_rng(0).normal(size=batch + (order, group.dof()))
    factors = np.asarray([math.factorial(k) for k in range(order)])[:, None]
    coefficients = ordinary / factors
    value = mr.CMVector(ordinary)
    for hat_name, vee_name, block_hat, block_size in [
        ("hat", "vee", group.hat, group.mat_size()),
        ("hat_adj", "vee_adj", group.hat_adj, group.mat_adj_size()),
    ]:
        expected = np.zeros(batch + (order * block_size, order * block_size))
        for row in range(order):
            for col in range(row + 1):
                expected[
                    ..., row*block_size:(row+1)*block_size,
                    col*block_size:(col+1)*block_size,
                ] = block_hat(coefficients[..., row-col, :])
        actual = getattr(mr.CMTM, hat_name)(group, value)
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        recovered = getattr(mr.CMTM, vee_name)(group, actual)
        assert isinstance(recovered, mr.CMVector)
        np.testing.assert_allclose(recovered.cm_vecs(), coefficients)
        np.testing.assert_allclose(recovered.vecs(), ordinary)
        # Array compatibility must not divide already-normalized values again.
        legacy = getattr(mr.CMTM, hat_name)(group, coefficients)
        np.testing.assert_allclose(legacy, expected)
        np.testing.assert_allclose(
            getattr(mr.CMTM, vee_name)(group, legacy, as_array=True), coefficients
        )


@pytest.mark.parametrize("group", [mr.SO3, mr.SE3, mr.SE3wrench])
def test_cm_adjoint_commute_identity(group):
    rng = np.random.default_rng(1)
    a, b = (mr.CMVector(rng.normal(size=(2, 5, group.dof()))) for _ in range(2))
    left = mr.CMTM.hat_adj(group, a) @ b.cm_vec()[..., None]
    right = mr.CMTM.hat_commute_adj(group, b) @ a.cm_vec()[..., None]
    np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("batch", [(), (2,)])
def test_se3_hat_commute_rectangular_blocks(batch):
    ordinary = np.random.default_rng(2).normal(size=batch + (5, 4))
    value = mr.CMVector(ordinary)
    actual = mr.CMTM.hat_commute(mr.SE3, value)
    assert actual.shape == batch + (20, 30)
    expected = np.zeros_like(actual)
    for row in range(5):
        for col in range(row + 1):
            expected[..., 4*row:4*(row+1), 6*col:6*(col+1)] = mr.SE3.hat_commute(
                value.cm_vecs()[..., row-col, :]
            )
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(mr.CMTM.hat_cm_commute(mr.SE3, value), expected)


def test_cm_compatibility_names():
    value = mr.CMVector(np.random.default_rng(3).normal(size=(5, 6)))
    for canonical, alias in [
        ("hat", "hat_cm"), ("hat_adj", "hat_cm_adj"),
        ("hat_commute_adj", "hat_cm_commute_adj"),
    ]:
        np.testing.assert_array_equal(
            getattr(mr.CMTM, canonical)(mr.SE3, value),
            getattr(mr.CMTM, alias)(mr.SE3, value),
        )
    for hat, alias in [("hat", "vee_cm"), ("hat_adj", "vee_cm_adj")]:
        recovered = getattr(mr.CMTM, alias)(mr.SE3, getattr(mr.CMTM, hat)(mr.SE3, value))
        np.testing.assert_allclose(recovered.vecs(), value.vecs())


@pytest.mark.parametrize("group", [mr.SO3, mr.SE3])
@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("input_dtype, result_dtype", [
    (np.float32, np.float32),
    (np.float64, np.float64),
    (np.int32, np.float64),
    (np.int64, np.float64),
    (np.complex64, np.complex64),
    (np.complex128, np.complex128),
])
def test_vee_return_dtype_averaging_and_ownership(group, adjoint, input_dtype, result_dtype):
    hat = group.hat_adj if adjoint else group.hat
    size = group.mat_adj_size() if adjoint else group.mat_size()
    canonical = mr.CMTM.vee_adj if adjoint else mr.CMTM.vee
    alias = mr.CMTM.vee_cm_adj if adjoint else mr.CMTM.vee_cm
    order = 4
    coefficients = np.arange(1, order * group.dof() + 1).reshape(order, group.dof()).astype(input_dtype)
    if np.issubdtype(input_dtype, np.complexfloating):
        coefficients += 1j * coefficients
    matrix = np.zeros((2, order * size, order * size), dtype=input_dtype)
    expected = np.empty((2, order, group.dof()), dtype=result_dtype)
    # Deliberately distinct blocks on each subdiagonal verify averaging;
    # integer inputs must preserve fractional averages such as 1.5.
    for offset in range(order):
        expected[:, offset, :] = coefficients[offset] + (order - offset - 1) / 2
        for col in range(order - offset):
            row = col + offset
            matrix[:, row*size:(row+1)*size, col*size:(col+1)*size] = hat(coefficients[offset] + col)
    snapshot = matrix.copy()
    recovered = canonical(group, matrix)
    compatibility = alias(group, matrix)
    array = canonical(group, matrix, as_array=True)
    assert isinstance(recovered, mr.CMVector)
    assert isinstance(compatibility, mr.CMVector)
    assert isinstance(array, np.ndarray)
    for output in (recovered.cm_vecs(), compatibility.cm_vecs(), array):
        assert output.dtype == np.dtype(result_dtype)
        np.testing.assert_allclose(output, expected)
        assert not np.shares_memory(output, matrix)
    factors = np.array([1, 1, 2, 6])[:, None]
    np.testing.assert_allclose(recovered.vecs(), expected * factors)

    recovered.cm_vecs()[...] = 0
    array[...] = -1
    np.testing.assert_array_equal(matrix, snapshot)
    np.testing.assert_array_equal(compatibility.cm_vecs(), expected)
    matrix[...] = 0
    np.testing.assert_array_equal(compatibility.cm_vecs(), expected)
