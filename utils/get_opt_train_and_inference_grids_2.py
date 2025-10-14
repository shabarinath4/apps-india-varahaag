import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
from tqdm import tqdm
import h3
import numpy as np
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
from kneed import KneeLocator
from tqdm import tqdm

H3_RES = 2  # ~500 km hex grid

print("Loading CSV...")
df = pd.read_csv(r"C:\Users\91814\Downloads\all_datasets_summary.csv")
print(f"Data loaded: {len(df)} rows, columns: {df.columns.tolist()}")

target_crops = ['corn','maize','sunflower','rapeseed','sorghum',
                'millet','wheat','barley','rye','soybean']

def normalize_crop(label):
    label = str(label).lower()
    for crop in target_crops:
        if crop in label:
            return crop
    return 'others'

df['sampling_label'] = df['sampling_label'].apply(normalize_crop)
df['cluster_id'] = -1

print("Assigning H3 indices...")
coords = df[['lat','lon']].to_numpy()
print(dir(h3))

df['h3_index'] = [h3.latlng_to_cell(lat, lon, H3_RES) for lat, lon in coords]

for crop in tqdm(df['sampling_label'].unique(), desc="Processing crops"):
    print(f"\n=== Processing crop: {crop} ===")
    crop_idx = df['sampling_label'] == crop
    crop_points = df.loc[crop_idx, ['lat','lon']].to_numpy()
    n_points = len(crop_points)
    print(f"Number of points: {n_points}")

    if n_points < 5:
        df.loc[crop_idx, 'cluster_id'] = 0
        print(f"Too few points, assigned cluster 0")
        continue

    # Aggregate points by H3 grid
    grid_df = df.loc[crop_idx].groupby('h3_index').agg(
        lat_center=('lat', 'mean'),
        lon_center=('lon', 'mean'),
        point_count=('lat', 'count')
    ).reset_index()

    print(f"Number of grids for crop '{crop}': {len(grid_df)}")

    # Normalize features for clustering
    X = grid_df[['lat_center','lon_center','point_count']].to_numpy()
    X[:,2] = X[:,2] / X[:,2].max()  # scale point_count to 0-1

    # Determine number of clusters using KMeans with KneeLocator
    inertias = []
    K_range = range(2, min(50, len(X)//2)+1)
    for k in K_range:
        kmeans_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans_tmp.fit(X)
        inertias.append(kmeans_tmp.inertia_)

    from kneed import KneeLocator
    knee = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
    best_k = knee.knee if knee.knee is not None else max(10, len(X))
    best_k = max(25, best_k)
    print(f"Best K determined: {best_k}")

    # Fit KMeans on grids
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    grid_df['cluster'] = kmeans.fit_predict(X)

    # Assign cluster to each original point based on its H3 grid
    h3_to_cluster = dict(zip(grid_df['h3_index'], grid_df['cluster']))
    df.loc[crop_idx, 'cluster_id'] = df.loc[crop_idx, 'h3_index'].map(h3_to_cluster)

    print(f"Clusters assigned, unique clusters: {len(grid_df['cluster'].unique())}")

    # Save shapefile for crop
    crop_gdf = gpd.GeoDataFrame(df.loc[crop_idx].copy(),
                                geometry=gpd.points_from_xy(df.loc[crop_idx,'lon'],
                                                            df.loc[crop_idx,'lat']),
                                crs="EPSG:4326")
    crop_gdf.to_file(f"{crop}_clusters.shp")
    print(f"Saved shapefile for crop '{crop}'")

# ---------------- Grid creation ----------------
grid_size_deg = 10 / 111  # ~10 km grid
lat_min, lat_max = df['lat'].min(), df['lat'].max()
lon_min, lon_max = df['lon'].min(), df['lon'].max()
lat_edges = np.arange(lat_min, lat_max + grid_size_deg, grid_size_deg)
lon_edges = np.arange(lon_min, lon_max + grid_size_deg, grid_size_deg)

grid_cells = []
grid_id = 0
for i in range(len(lat_edges) - 1):
    for j in range(len(lon_edges) - 1):
        poly = Polygon([
            (lon_edges[j], lat_edges[i]),
            (lon_edges[j + 1], lat_edges[i]),
            (lon_edges[j + 1], lat_edges[i + 1]),
            (lon_edges[j], lat_edges[i + 1])
        ])
        grid_cells.append({'grid_id': grid_id, 'geometry': poly})
        grid_id += 1

grid_gdf = gpd.GeoDataFrame(grid_cells, crs="EPSG:4326")

# ---------------- Assign points to grids ----------------
points_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
joined = gpd.sjoin(points_gdf, grid_gdf, how='left', predicate='within')

# ---------------- Compute grid stats ----------------
grid_stats = joined.groupby('grid_id').agg(
    density=('geometry', 'count'),
    crop_count=('sampling_label', lambda x: x.nunique()),
    unique_clusters=('cluster_id', lambda x: len(np.unique(x)))
).reset_index()
grid_stats = grid_stats.merge(grid_gdf[['grid_id', 'geometry']], on='grid_id')
grid_stats_gdf = gpd.GeoDataFrame(grid_stats, geometry='geometry', crs="EPSG:4326")
grid_stats_gdf = grid_stats_gdf.sort_values(['unique_clusters', 'crop_count', 'density'], ascending=False)

# ---------------- Greedy selection ----------------
clusters_needed = set((row['sampling_label'], row['cluster_id']) for _, row in df.iterrows())
covered_clusters = set()
selected_grids = []

for _, row in tqdm(grid_stats_gdf.iterrows(), total=len(grid_stats_gdf)):
    grid_id = row['grid_id']
    grid_points = joined[joined['grid_id'] == grid_id]
    clusters_in_grid = set((r['sampling_label'], r['cluster_id']) for _, r in grid_points.iterrows())
    new_clusters = clusters_in_grid - covered_clusters
    if new_clusters:
        selected_grids.append(grid_id)
        covered_clusters |= new_clusters
    if covered_clusters == clusters_needed:
        break

selected_grid_gdf = grid_stats_gdf[grid_stats_gdf['grid_id'].isin(selected_grids)].copy()
selected_grid_gdf = selected_grid_gdf[['grid_id', 'geometry', 'unique_clusters', 'crop_count', 'density']]
selected_grid_gdf.to_file("selected_grids.shp")

# ---------------- Save points in selected grids ----------------
records = []
for grid_id in selected_grids:
    grid_points = joined[joined['grid_id'] == grid_id]
    for _, r in grid_points.iterrows():
        records.append({
            'crop': r['sampling_label'],
            'cluster': r['cluster_id'],
            'grid_id': grid_id,
            'num_points_in_grid': len(grid_points)
        })
pd.DataFrame(records).to_csv("selected_grids_points.csv", index=False)