import numpy as np
import pytest

import mathrobo as mr


@pytest.mark.parametrize("digit", [1, 2, 3, 5, 8])
def test_gq_integrate_exact_for_polynomials(digit):
    degree = 2 * digit - 1
    result = mr.gq_integrate(lambda x: np.array([x**degree, 1.0]), -0.5, 2.0, digit=digit)
    expected = (2.0**(degree + 1) - (-0.5)**(degree + 1)) / (degree + 1)
    np.testing.assert_allclose(result, [expected, 2.5], rtol=1e-12)


def test_gq_integrate_rejects_invalid_digit():
    with pytest.raises(ValueError, match="digit"):
        mr.gq_integrate(np.exp, 0.0, 1.0, digit=0)
