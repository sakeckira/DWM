# import numpy as np
# import matplotlib.pyplot as plt

# points = np.array([
#     [1, 2], [1.5, 1.8], [5, 8],
#     [8, 8], [1, 0.6], [9, 11],
#     [8, 2], [10, 2], [9, 3]
# ])


# k = 2

# np.random.seed(42)
# centroids = points[np.random.choice(len(points), k, replace=False)]

# def euclidian(a,b):
#     return np.sqrt(np.sum((a-b)**2))

# prev_assignments = None
# for iteration in range(100):
#     clusters = [[] for _ in range(k)]
#     assignments = []

#     for point in points:
#         distances = [euclidian(point, centroid) for centroid in centroids]
#         cluster_index = np.argmin(distances)
#         clusters[cluster_index].append(point)
#         assignments.append(cluster_index)

#     assignments = np.array(assignments)

#     if prev_assignments is not None and np.all(assignments == prev_assignments):
#         print(f"Converged at iteration {iteration}")
#         break
#     prev_assignments= assignments

#     for i in range(k):
#         if clusters[i]:
#             centroids[i] = np.mean(clusters[i], axis=0)

# colors = ['r', 'b']
# for i in range(k):
#     cluster = np.array(clusters[i])
#     plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], label=f'Cluster {i}')
# plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='X', label='Centroids')
# plt.title("K-Means Clustering (2D)")
# plt.legend()
# plt.grid(True)
# plt.show()



import pandas as pd
import math
import random
import matplotlib.pyplot as plt

# Step 1: Load 2D data from Excel

df = pd.read_excel(r'/content/K_Means_2D.xlsx')
data = df.values.tolist()  # Each row is a [x, y] point
# Convert lists to tuples to match the first code
data = [tuple(point) for point in data]

# Step 2: Get user input for number of clusters
k = int(input("Enter the number of clusters (k): "))
if k > len(data):
    print(f"Error: k ({k}) cannot be larger than number of points ({len(data)})")
    exit()

# Step 3: Choose initial centroids (first and third points for k=2, else random)
if k == 2 and len(data) >= 3:
    centroids = [data[0], data[2]]  # First and third points: (2.0, 2.0) and (1.0, 1.0)
    print("Initial centroids (first and third points):", centroids)
else:
    centroids = random.sample(data, k)
    print("Initial centroids:", centroids)

# Step 4: Function to calculate Euclidean distance
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Step 5: Assign each point to the nearest centroid
def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean(point, centroid) for centroid in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(point)
    return clusters

# Step 6: Update centroids as the mean of each cluster
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            x_coords = [p[0] for p in cluster]
            y_coords = [p[1] for p in cluster]
            new_centroids.append((sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords)))
        else:
            new_centroids.append((0, 0))  # Default for empty cluster
    return new_centroids

# Step 7: Repeat until convergence or max iterations
def are_centroids_equal(c1, c2, tolerance=1e-6):
    if len(c1) != len(c2):
        return False
    for p1, p2 in zip(c1, c2):
        if not (abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance):
            return False
    return True

for iteration in range(10):
    clusters = assign_clusters(data, centroids)
    new_centroids = update_centroids(clusters)
    if are_centroids_equal(new_centroids, centroids):
        print(f"Converged at iteration {iteration+1}!")
        break
    centroids = new_centroids

# Step 8: Validate all points are clustered
all_clustered_points = [p for cluster in clusters for p in cluster]
missing_points = [p for p in data if p not in all_clustered_points]
if missing_points:
    print("Error: Missing points:", missing_points)
else:
    print("All points assigned.")

# Step 9: Print final clusters
print("\nFinal clusters:")
for idx, cluster in enumerate(clusters, 1):
    print(f"Cluster {idx} ({len(cluster)} points): {cluster}")

# Step 10: Print final centroids
print("\nFinal centroids:")
for idx, centroid in enumerate(centroids, 1):
    print(f"Centroid {idx}: {centroid}")

# Step 11: Visualize
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
plt.figure(figsize=(8, 6))
for i, cluster in enumerate(clusters):
    if cluster:
        x = [p[0] for p in cluster]
        y = [p[1] for p in cluster]
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'Cluster {i+1}', s=100)
        for point in cluster:
            plt.text(point[0], point[1], f"{point}", fontsize=10, ha='right')

# Plot centroids
mean_x = [m[0] for m in centroids]
mean_y = [m[1] for m in centroids]
plt.scatter(mean_x, mean_y, color='black', marker='x', s=200, label='Centroids')

plt.legend()
plt.title(f"2D K-Means Clustering (k={k})")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
