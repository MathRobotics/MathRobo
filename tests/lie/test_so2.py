import numpy as np
import pytest
import mathrobo.lie.so2 as so2_module
import mathrobo as mr


def test_so2_set_adj_identity():
        np.testing.assert_array_equal(mr.SO2.eye().mat_adj(), np.ones((1, 1)))
        with pytest.raises(ValueError, match="cannot determine"):
                mr.SO2.set_mat_adj(np.ones((1, 1)))


def test_so2_is_exported_from_top_level_package():
        assert mr.SO2 is so2_module.SO2


def test_lie_matmul_rejects_unsupported_operands():
        with pytest.raises(TypeError):
                so2_module.SO2.eye() @ object()

        with pytest.raises(TypeError):
                mr.SO3.eye() @ object()

        with pytest.raises(TypeError):
                mr.SE3.eye() @ object()
