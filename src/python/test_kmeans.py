from python.kmeans import go
import numpy as np


def test_kmeans_basic():
    np.testing.assert_equal(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]), go())
