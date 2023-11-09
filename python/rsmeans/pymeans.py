from typing import TypeVar

import numpy as np
from nptyping import NDArray, Float, Int, Shape
from dataclasses import dataclass
from sklearn.metrics import pairwise_distances

R = TypeVar("R")
Data = NDArray[Shape["R, C"], Float]
Centroids = NDArray[Shape["K, C"], Float]
Assignments = NDArray[Shape["R"], Int]


@dataclass
class State:
    centroids: Centroids  # shape: (num_clusters, features)
    cluster_assignments: Assignments  # shape: (samples, )


def generate_data(rows, cols) -> Data:
    data = np.random.random(size=(rows, cols))
    data[: rows // 3, 0] += 10
    data[2 * rows // 3:, 1] += 10
    return data


def assign_to_clusters(data: Data, state: State, sklearn_pairwise=True):
    if sklearn_pairwise:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        distances = pairwise_distances(data, state.centroids, metric="sqeuclidean",
                                       force_all_finite="allow-nan", n_jobs=-1)
    else:
        distances = np.zeros((len(data), len(state.centroids)))
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                distances[i, j] = euclidean_dist(data[i, :], state.centroids[j, :])


    nearest = distances.argmin(axis=1)
    state.cluster_assignments = nearest


def euclidean_dist(x, y):
    return ((x - y) ** 2).sum()


def update(data: Data, state: State, sklearn_pairwise=True):
    for _ in range(5):
        assign_to_clusters(data, state, sklearn_pairwise=sklearn_pairwise)
        new_centroids = np.zeros_like(state.centroids)
        for cluster in range(state.centroids.shape[0]):
            # TODO we could end up with no items in one of the clusters, so we're taking the mean of an empty slice
            new_centroids[cluster, :] = data[
                                        state.cluster_assignments == cluster, :
                                        ].mean(axis=0)
        state.centroids = new_centroids


def init_state(data: Data, k) -> State:
    n = len(data)
    indices = [0, n // 2, n - 1] if k == 3 else np.unique(np.random.randint(low=0, high=n, size=(k * 2,)))[:k]
    return State(data[indices, :], np.zeros((n,), dtype=np.int64))


def cluster_numpy(data, k, sklearn_pairwise=True):
    state = init_state(data, k)
    update(data, state, sklearn_pairwise=sklearn_pairwise)
    return state.cluster_assignments


if __name__ == "__main__":
    cluster_numpy(generate_data(10, 2), k=3)
