import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reproducible_random_samples():
    np.random.seed(2026)
