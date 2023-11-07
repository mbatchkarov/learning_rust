from collections import Counter

import pytest

from rsmeans import cluster as cluster_rust
from rsmeans.pymeans import cluster_numpy, generate_data
from cmeans.pywrapper import cluster as cluster_c
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

SMALL_DATA = generate_data(10, 5)
EXPECTED = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
K = 3


def test_kmeans_python():
    np.testing.assert_equal(EXPECTED, cluster_numpy(SMALL_DATA, K))


def test_kmeans_rust():  # run `maturin develop` or `maturin build -r --strip` to build the rust extension
    np.testing.assert_equal(EXPECTED, cluster_rust(SMALL_DATA, K))


def test_kmeans_c():  # run `make sharedlib` to build the C extension
    np.testing.assert_equal(EXPECTED, cluster_c(SMALL_DATA, K))


def cluster_sklearn(data, k):
    return KMeans(n_clusters=k, init='random', max_iter=5, n_init=1, tol=1e-10).fit_predict(data)


@pytest.mark.parametrize("cluster_func", [cluster_numpy, cluster_rust, cluster_c, cluster_sklearn])
def test_larger_data(cluster_func):
    k = 10
    big_data, gold_labels = make_blobs(n_samples=10_000, n_features=150, centers=k)
    print(Counter(gold_labels))
    predicted = cluster_func(big_data, k)
    print(Counter(predicted))
