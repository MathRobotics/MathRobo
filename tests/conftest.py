"""Make tests that draw from NumPy's global RNG reproducible.

Each test seeds ``np.random`` from its node id, so results do not depend on test
order or selection. Set ``MATHROBO_TEST_SEED`` to another integer to explore
different inputs; the seed in use is reported in the header and on failure.
"""
import os
import zlib

import numpy as np
import pytest

BASE_SEED = int(os.environ.get("MATHROBO_TEST_SEED", "0"))


def pytest_report_header(config):
    return f"MATHROBO_TEST_SEED={BASE_SEED}"


@pytest.fixture(autouse=True)
def _seed_numpy_global_rng(request):
    seed = (zlib.crc32(request.node.nodeid.encode()) + BASE_SEED) % 2**32
    np.random.seed(seed)
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and report.failed:
        report.sections.append(("random seed", f"MATHROBO_TEST_SEED={BASE_SEED}"))
