from scipy.spatial.distance import cdist
import cv2
import numpy as np
from kneed import KneeLocator

class Score:
    def rgb_to_hsv(self, bgr_array):
        hsv_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2HSV)
        return hsv_array

    def inter_cluster_dist(self, centroids):
        dist_matrix = cdist(centroids, centroids, metric='euclidean')
        sum_of_distances = dist_matrix.sum(axis=1)
        sorted_indices = np.argsort(-sum_of_distances)
        sorted_distances = sum_of_distances[sorted_indices]
        sorted_clusters = centroids[sorted_indices]
        cluster_dist = dict(zip(map(tuple, sorted_clusters), sorted_distances))
        return cluster_dist

    def cluster_stats(self, centroids):
        max_per_dim = centroids.max(axis=0)
        return max_per_dim

    def num_clusters_elbow(self, cluster_scores):
        clusters = sorted(cluster_scores.keys())
        scores = [cluster_scores[c] for c in clusters]
        knee = KneeLocator(clusters, scores, curve='convex', direction='decreasing')
        num_of_clusters = knee.knee
        assigned_clusters = [c for c in clusters if c <= num_of_clusters]
        return num_of_clusters, assigned_clusters
    
    def num_clusters_coverage(self, cluster_array, sorted_cluster_dict, threshold=0.9):
        total_samples = cluster_array.shape[0]
        covered_samples = np.zeros(total_samples, dtype=bool)
        assigned_clusters = []
        cumulative_coverage = 0.0

        for cluster in sorted_cluster_dict.keys():
            cluster_mask = (cluster_array == cluster).any(axis=1)
            newly_covered = cluster_mask & ~covered_samples
            covered_samples |= newly_covered
            assigned_clusters.append(cluster)
            cumulative_coverage = covered_samples.mean()
            if cumulative_coverage >= threshold:
                break
        num_of_clusters = len(assigned_clusters)
        return num_of_clusters, assigned_clusters


