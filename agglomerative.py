import numpy as np
import matplotlib.pyplot as plt

# Sample points
points = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11],
    [8, 2], [10, 2], [9, 3]
])

# Euclidean distance
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Average linkage distance between clusters
def average_linkage(c1, c2):
    return np.min([euclidean(p1, p2) for p1 in c1 for p2 in c2])

def average_linkage1(c1, c2):
    total_dist = 0
    count = 0
    for p1 in c1:
        for p2 in c2:
            total_dist += euclidean(p1, p2)
            count += 1
    return total_dist / count if count > 0 else float('inf')

# Start with each point as its own cluster
clusters = [[p] for p in points]
target_k = 2

# Merge clusters until only `target_k` clusters remain
while len(clusters) > target_k:
    min_distance = float('inf')  # start with the largest possible number
    closest_pair = (0, 1)        # just a placeholder

    # Go through every pair of clusters
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):  # don't repeat pairs
            # Get the two clusters
            cluster1 = clusters[i]
            cluster2 = clusters[j]
            
            # Calculate how far apart they are (average linkage)
            dist = average_linkage(cluster1, cluster2)
            
            # If this pair is closer than our current best, update
            if dist < min_distance:
                min_distance = dist
                closest_pair = (i, j)

    # After the loop, we now know the best pair to merge:
    i, j = closest_pair

    clusters.append(clusters[i] + clusters[j])
    clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]

# Plot results
colors = ['r', 'b', 'g', 'c', 'm']

for i in range(len(clusters)):
    x = [p[0] for p in clusters[i]]
    y = [p[1] for p in clusters[i]]
    plt.scatter(x, y, color=colors[i], label=f'Cluster {i}')

plt.title("Agglomerative Clustering (k=2)")
plt.legend()
plt.grid(True)
plt.show()
