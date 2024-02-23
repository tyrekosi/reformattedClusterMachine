import hashlib
import time
import numpy as np
import os
import json
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from typing import Dict, Tuple, List
from utils.toDatasheet import toDatasheet

def calculate_cluster_metrics(coordinates, cluster_labels):
    cluster_metrics = {}
    for label in set(cluster_labels):
        if label == -1:  # skip noise
            continue
        indexes = np.where(cluster_labels == label)[0]
        cluster_points = np.array([list(coordinates.values())[i] for i in indexes])
        dist_matrix = pairwise_distances(cluster_points, cluster_points)
        avg_distance = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        farthest_pairs = np.unravel_index(np.argsort(dist_matrix, axis=None)[-5:], dist_matrix.shape)
        cluster_metrics[str(label)] = {
            'average_distance': float(avg_distance),
            'farthest_pairs': [{'pair': [list(coordinates.keys())[indexes[i]], list(coordinates.keys())[indexes[j]]], 'distance': float(dist_matrix[i, j])} for i, j in zip(*farthest_pairs) if i != j][:5]
        }
    return cluster_metrics

def create_cluster_labels_array(coordinates, clusters):
    """
    Create an array of cluster labels based on the cluster membership.
    """
    cluster_labels = np.full(shape=len(coordinates), fill_value=-1)
    for label, members in clusters.items():
        for member in members:
            idx = list(coordinates.keys()).index(member)
            cluster_labels[idx] = label
    return cluster_labels

def dbscan_clustering(coordinates: Dict[str, List[float]], eps: float = 0.05, min_samples: int = 3) -> Tuple[Dict[int, List[str]], Dict[str, Dict]]:
    coords_array = np.array(list(coordinates.values()))
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_array)
    clusters = {label: [] for label in set(clustering.labels_) if label != -1}
    for idx, label in enumerate(clustering.labels_):
        if label != -1:
            clusters[label].append(list(coordinates.keys())[idx])
    cluster_metrics = calculate_cluster_metrics(coordinates, clustering.labels_)
    return clusters, cluster_metrics

def integrate_small_clusters(coordinates: Dict[str, List[float]], clusters: Dict[int, List[str]], eps: float, min_cluster_size: int = 2) -> Dict[int, List[str]]:
    coords_array = np.array(list(coordinates.values()))
    cluster_labels = np.full(shape=len(coords_array), fill_value=-1)
    large_clusters = {k: v for k, v in clusters.items() if len(v) >= min_cluster_size}
    small_clusters = {k: v for k, v in clusters.items() if len(v) < min_cluster_size}
    id_to_index = {id: index for index, id in enumerate(coordinates.keys())}

    # Calculate midpoints and average distances for large clusters
    large_cluster_midpoints = {}
    large_cluster_avg_distances = {}
    for label, members in large_clusters.items():
        member_coords = np.array([coordinates[member] for member in members])
        large_cluster_midpoints[label] = np.mean(member_coords, axis=0)
        dist_matrix = pairwise_distances(member_coords, member_coords)
        avg_distance = np.mean(dist_matrix[np.triu_indices_from(dist_matrix, k=1)])
        large_cluster_avg_distances[label] = avg_distance

    for label, members in small_clusters.items():
        member_coords = np.array([coordinates[member] for member in members])
        small_cluster_midpoint = np.mean(member_coords, axis=0)

        # Find the nearest large cluster
        nearest_large_cluster = None
        min_distance_to_large = float('inf')
        for large_label, large_midpoint in large_cluster_midpoints.items():
            distance = np.linalg.norm(small_cluster_midpoint - large_midpoint)
            if distance < min_distance_to_large:
                min_distance_to_large = distance
                nearest_large_cluster = large_label

        # Merge if the conditions are met
        if nearest_large_cluster is not None and min_distance_to_large < 1.1 * large_cluster_avg_distances[nearest_large_cluster]:
            for member in members:
                idx = id_to_index[member]
                cluster_labels[idx] = nearest_large_cluster
                clusters[nearest_large_cluster].append(member)

    # Adjust cluster labels for large clusters
    for label, members in large_clusters.items():
        for member in members:
            idx = id_to_index[member]
            cluster_labels[idx] = label

    # Compile adjusted clusters
    adjusted_clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label != -1:
            if label not in adjusted_clusters:
                adjusted_clusters[label] = []
            adjusted_clusters[label].append(list(coordinates.keys())[idx])

    return {k: v for k, v in adjusted_clusters.items() if v}

def output_data(coordinates, clusters, cluster_metrics, base_dir='clusters'):
    run_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:10]  
    run_dir = os.path.join(base_dir, run_hash)
    
    clusters_subdir = os.path.join(run_dir, "clusters")  
    info_path = os.path.join(run_dir, 'cluster_info.json')  
    os.makedirs(clusters_subdir, exist_ok=True)
    
    for label, identifiers in clusters.items():
        cluster_file_path = os.path.join(clusters_subdir, f'cluster_{label}.txt')
        with open(cluster_file_path, 'w') as f:
            for iden in identifiers:
                f.write(f"{iden},{','.join(map(str, coordinates[iden]))}\n")
    
    with open(info_path, 'w') as f:
        json.dump(cluster_metrics, f, indent=4)

    toDatasheet(clusters_subdir, run_dir, run_hash)