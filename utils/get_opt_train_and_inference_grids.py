import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------
# Step 0: Assume your DataFrame
# ----------------------------
df = pd.read_csv(r"C:\Users\91814\Downloads\all_datasets_summary.csv")

# ----------------------------
# Step 1: Cluster within each crop (if needed)
# ----------------------------
crop_clusters = {}
for crop in df['sampling_label'].unique():
    print('crop', crop)
    crop_points = df[df['sampling_label']==crop][['lat','lon']].values
    n_clusters = min(5, len(crop_points)//10)  # choose cluster number based on points
    if n_clusters <= 1:
        # small crop, just take the mean
        densest_points = [crop_points.mean(axis=0)]
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(crop_points)
        densest_points = []
        for cluster_id in np.unique(labels):
            cluster_points = crop_points[labels==cluster_id]
            centroid = cluster_points.mean(axis=0)
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            densest_idx = np.argmin(dists)
            densest_points.append(cluster_points[densest_idx])
    crop_clusters[crop] = np.array(densest_points)

# ----------------------------
# Step 2: Prepare candidates
# ----------------------------
candidates = []
for crop, reps in crop_clusters.items():
    for rep in reps:
        candidates.append({'crop': crop, 'coord': rep})

coords = np.array([c['coord'] for c in candidates])
coords_rad = np.deg2rad(coords)

# ----------------------------
# Step 3: Build BallTree
# ----------------------------
tree = BallTree(coords_rad, metric="haversine")
radius_km = 50  # choose suitable radius

# ----------------------------
# Step 4: Greedy cross-crop selection
# ----------------------------
uncovered = set((c['crop'], tuple(c['coord'])) for c in candidates)
selected = []

while uncovered:
    best_point, best_cover = None, set()
    for i, cand in enumerate(candidates):
        ckey = (cand['crop'], tuple(cand['coord']))
        if ckey not in uncovered:
            continue
        neighbors_idx = tree.query_radius([coords_rad[i]], r=radius_km/6371.0)[0]
        cover = set((candidates[j]['crop'], tuple(candidates[j]['coord'])) for j in neighbors_idx)
        new_cover = cover & uncovered
        if len(new_cover) > len(best_cover):
            best_point = cand
            best_cover = new_cover
    selected.append(best_point)
    uncovered -= best_cover

# ----------------------------
# Step 5: Visualization
# ----------------------------
plt.figure(figsize=(10,6))
colors = {crop:plt.cm.tab10(i) for i,crop in enumerate(df['sampling_label'].unique())}

for crop in df['sampling_label'].unique():
    points = df[df['sampling_label']==crop][['lat','lon']].values
    plt.scatter(points[:,1], points[:,0], s=20, color=colors[crop], alpha=0.3, label=f"{crop}")

for sel in selected:
    lat, lon = sel['coord']
    plt.scatter(lon, lat, color='red', s=100, edgecolor='black', label=f"selected rep")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Multi-crop clusters and selected representatives")
plt.legend()
plt.show()

# ----------------------------
# Step 6: Print selected points
# ----------------------------
print("Selected representative points:")
for sel in selected:
    print(f"{sel['crop']} -> {sel['coord']}")

exit()
