import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('data.txt', delimiter=',')[:, :2]
centroids = np.genfromtxt('centroids.txt', delimiter=',')[:, :2]
clusters = np.genfromtxt('clusters.txt', delimiter=',')[:-1]
print(data.shape, centroids.shape, clusters.shape)

LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in clusters]

plt.scatter(data[:, 0], data[:, 1], c=label_color)
plt.scatter(centroids[:, 0], centroids[:, 1], c='k')

plt.savefig('data.png')
