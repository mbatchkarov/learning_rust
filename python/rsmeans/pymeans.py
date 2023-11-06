from typing import TypeVar

import numpy as np
from nptyping import NDArray, Float, Int, Shape
from dataclasses import dataclass

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


def assign_to_clusters(data: Data, state: State):
    distances = np.zeros((len(data), len(state.centroids)))
    for i in range(distances.shape[0]):
        for j in range(distances.shape[1]):
            distances[i, j] = euclidean_dist(data[i, :], state.centroids[j, :])
    nearest = distances.argmin(axis=1)
    state.cluster_assignments = nearest


def euclidean_dist(x, y):
    return ((x - y) ** 2).sum()


def update(data: Data, state: State):
    for _ in range(5):
        assign_to_clusters(data, state)
        new_centroids = np.zeros_like(state.centroids)
        for cluster in range(state.centroids.shape[0]):
            new_centroids[cluster, :] = data[
                                        state.cluster_assignments == cluster, :
                                        ].mean(axis=0)
        state.centroids = new_centroids


def init_state(data: Data, k) -> State:
    n = len(data)
    indices = [0, n // 2, n - 1] if k == 3 else np.random.randint(low=0, high=n, size=(k,))
    return State(data[indices, :], np.zeros((n,), dtype=np.int64))


def cluster_numpy(data, k):
    state = init_state(data, k)
    update(data, state)
    return state.cluster_assignments


if __name__ == "__main__":
    cluster_numpy(generate_data(10, 2), k=3)
