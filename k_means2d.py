import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11],
    [8, 2], [10, 2], [9, 3]
])


k = 2

np.random.seed(42)
centroids = points[np.random.choice(len(points), k, replace=False)]

def euclidian(a,b):
    return np.sqrt(np.sum((a-b)**2))

prev_assignments = None
for iteration in range(100):
    clusters = [[] for _ in range(k)]
    assignments = []

    for point in points:
        distances = [euclidian(point, centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
        assignments.append(cluster_index)

    assignments = np.array(assignments)

    if prev_assignments is not None and np.all(assignments == prev_assignments):
        print(f"Converged at iteration {iteration}")
        break
    prev_assignments= assignments

    for i in range(k):
        if clusters[i]:
            centroids[i] = np.mean(clusters[i], axis=0)

colors = ['r', 'b']
for i in range(k):
    cluster = np.array(clusters[i])
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', label='Centroids')
plt.title("K-Means Clustering (2D)")
plt.legend()
plt.grid(True)
plt.show()
