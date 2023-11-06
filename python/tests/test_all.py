import rsmeans
from rsmeans.pymeans import cluster_numpy, generate_data
import numpy as np


def test_sum_as_string():
    # rust module
    assert rsmeans.sum_as_string(1, 1) == "2"


def test_kmeans_python():
    np.testing.assert_equal(np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]), cluster_numpy())


def test_kmeans_rust():
    data = generate_data(10, 2)
    np.testing.assert_equal(
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]), rsmeans.cluster(data, 3)
    )
