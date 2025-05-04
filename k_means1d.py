import random

data = [1, 2 ,6,7,8,10,15,17,20]
k = 3       

centroids = random.sample(data,k)

def assign_clusters(data, centroids):
    clusters = {i: []  for i in range(k)}

    for point in data:
        distances = [abs(point-c) for c in centroids]
        closest_cluster = distances.index(min(distances))
        clusters[closest_cluster].append(point)
    return clusters

def update_centroids(clusters):
    new_centroids = []
    for point in clusters.values():
        new_centroids.append(sum(point)/len(point) if point else 0)
    return new_centroids

for iteration in range(100):
    clusters = assign_clusters(data, centroids)
    new_centroids = update_centroids(clusters)
    if new_centroids == centroids:
        break
    centroids = new_centroids


print("Final Centroids:", centroids)
for i, cluster in clusters.items():
    print(f"Cluster {i+1}: {cluster}")
