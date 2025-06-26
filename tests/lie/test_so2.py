import numpy as np
import mathrobo.lie.so2 as so2_module


def test_so2_set_adj_identity():
        rot = so2_module.SO2.set_mat_adj()
        np.testing.assert_array_equal(rot.mat(), np.eye(2))

