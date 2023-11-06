from datetime import datetime

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('data.txt', delimiter=',')[:, :-1]
print(data.shape)

start = datetime.now()

for _ in range(20):
    clusters = KMeans(n_clusters=3, init='random', max_iter=5, n_init=1, tol=1e-10).fit_predict(data)

print('Duration', datetime.now() - start)
LABEL_COLOR_MAP = {0: 'r', 1: 'g', 2: 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in clusters]

plt.scatter(data[:, 0], data[:, 1], c=label_color)

plt.savefig('data-sklearn.png')
