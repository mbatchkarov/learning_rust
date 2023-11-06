from rsmeans import cluster as cluster_rust
from rsmeans.pymeans import cluster_numpy, generate_data
from cmeans.pywrapper import cluster as cluster_c
import numpy as np

data = generate_data(10, 5)
expected = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
K = 3


def test_kmeans_python():
    np.testing.assert_equal(expected, cluster_numpy(data, K))


def test_kmeans_rust():  # run `maturin develop` to build the rust extension
    np.testing.assert_equal(expected, cluster_rust(data, K))


def test_kmeans_c():  # run `make sharedlib` to build the C extension
    np.testing.assert_equal(expected, cluster_c(data, K))
