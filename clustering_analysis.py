"""
Clustering Analysis for Fashion Runway Images
Implements multiple clustering methods and evaluation metrics
"""

import numpy as np
import pickle
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score,
    calinski_harabasz_score
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================

def reduce_dimensions(embeddings, n_components=50, method='pca'):
    """
    Reduce embedding dimensions using PCA
    
    Args:
        embeddings: (n_samples, n_features)
        n_components: target dimensionality
        method: 'pca' (can extend to UMAP, etc.)
    
    Returns:
        reduced_embeddings: (n_samples, n_components)
        reducer: fitted reducer object
    """
    print(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}...")
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        
        # Print explained variance
        explained_var = np.sum(reducer.explained_variance_ratio_)
        print(f"  Explained variance: {explained_var:.2%}")
        
    else:
        raise ValueError(f"Method {method} not supported")
    
    return reduced, reducer


# ============================================================================
# CLUSTERING ALGORITHMS
# ============================================================================

def kmeans_clustering(embeddings, k_values=[3, 5, 7, 10]):
    """
    Perform K-means clustering with different K values
    
    Args:
        embeddings: (n_samples, n_features)
        k_values: list of K values to try
    
    Returns:
        results: dict with clustering results for each K
    """
    results = {}
    
    for k in k_values:
        print(f"\nK-means with K={k}...")
        
        kmeans = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        sil_score = silhouette_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        
        results[k] = {
            'model': kmeans,
            'labels': labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        }
        
        print(f"  Silhouette Score: {sil_score:.3f}")
        print(f"  Davies-Bouldin Score: {db_score:.3f} (lower is better)")
        print(f"  Calinski-Harabasz Score: {ch_score:.1f} (higher is better)")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique, counts))}")
    
    return results


def hierarchical_clustering(embeddings, n_clusters_values=[3, 5, 7, 10]):
    """
    Perform Agglomerative (Hierarchical) clustering
    
    Args:
        embeddings: (n_samples, n_features)
        n_clusters_values: list of cluster counts to try
    
    Returns:
        results: dict with clustering results
    """
    results = {}
    
    for n_clusters in n_clusters_values:
        print(f"\nHierarchical clustering with {n_clusters} clusters...")
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = model.fit_predict(embeddings)
        
        # Calculate metrics
        sil_score = silhouette_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        
        results[n_clusters] = {
            'model': model,
            'labels': labels,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        }
        
        print(f"  Silhouette Score: {sil_score:.3f}")
        print(f"  Davies-Bouldin Score: {db_score:.3f}")
        print(f"  Calinski-Harabasz Score: {ch_score:.1f}")
    
    return results


def dbscan_clustering(embeddings, eps_values=[0.5, 1.0, 1.5], min_samples=5):
    """
    Perform DBSCAN clustering (density-based)
    Good for finding outliers
    
    Args:
        embeddings: (n_samples, n_features)
        eps_values: list of epsilon values (neighborhood size)
        min_samples: minimum samples per cluster
    
    Returns:
        results: dict with clustering results
    """
    results = {}
    
    for eps in eps_values:
        print(f"\nDBSCAN with eps={eps}, min_samples={min_samples}...")
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embeddings)
        
        # Count clusters (excluding noise points labeled -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        # Only calculate metrics if we have at least 2 clusters
        if n_clusters >= 2:
            # Remove noise points for metric calculation
            mask = labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(embeddings[mask], labels[mask])
                print(f"  Silhouette Score: {sil_score:.3f}")
            else:
                sil_score = None
        else:
            sil_score = None
        
        results[eps] = {
            'model': model,
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': sil_score
        }
    
    return results


# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================

def analyze_clusters_by_metadata(labels, metadata_df):
    """
    Analyze how clusters align with metadata (designer, season, etc.)
    
    Args:
        labels: cluster labels for each image
        metadata_df: DataFrame with columns ['designer', 'saison', 'annee']
    
    Returns:
        analysis: dict with cluster compositions
    """
    analysis = {}
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Skip noise in DBSCAN
            continue
        
        mask = labels == cluster_id
        cluster_metadata = metadata_df[mask]
        
        analysis[cluster_id] = {
            'size': mask.sum(),
            'designers': cluster_metadata['designer'].value_counts().to_dict(),
            'seasons': cluster_metadata['saison'].value_counts().to_dict(),
            'years': cluster_metadata['annee'].value_counts().to_dict()
        }
    
    return analysis


def compute_inter_intra_distances(embeddings, labels):
    """
    Compute inter-cluster and intra-cluster distances
    
    Args:
        embeddings: (n_samples, n_features)
        labels: cluster assignments
    
    Returns:
        metrics: dict with distance metrics
    """
    from scipy.spatial.distance import cdist
    
    unique_labels = np.unique(labels[labels != -1])  # Exclude noise
    
    # Compute cluster centroids
    centroids = []
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # Intra-cluster distances (average distance to centroid)
    intra_distances = []
    for label in unique_labels:
        cluster_points = embeddings[labels == label]
        centroid = centroids[label]
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        intra_distances.append(distances.mean())
    
    # Inter-cluster distances (pairwise centroid distances)
    if len(centroids) > 1:
        inter_distances = cdist(centroids, centroids, metric='euclidean')
        # Get upper triangle (excluding diagonal)
        inter_dist_values = inter_distances[np.triu_indices_from(inter_distances, k=1)]
    else:
        inter_dist_values = [0]
    
    metrics = {
        'mean_intra_distance': np.mean(intra_distances),
        'std_intra_distance': np.std(intra_distances),
        'mean_inter_distance': np.mean(inter_dist_values),
        'std_inter_distance': np.std(inter_dist_values),
        'separation_ratio': np.mean(inter_dist_values) / np.mean(intra_distances)
    }
    
    return metrics


def create_results_table(clustering_results):
    """
    Create a summary table of all clustering results
    
    Args:
        clustering_results: dict of results from different methods
    
    Returns:
        df: pandas DataFrame with comparison
    """
    rows = []
    
    for method_name, method_results in clustering_results.items():
        for k, result in method_results.items():
            row = {
                'Method': method_name,
                'K/Clusters': k,
                'Silhouette': result.get('silhouette', None),
                'Davies-Bouldin': result.get('davies_bouldin', None),
                'Calinski-Harabasz': result.get('calinski_harabasz', None)
            }
            
            if 'inertia' in result:
                row['Inertia'] = result['inertia']
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CLUSTERING ANALYSIS FOR FASHION RUNWAY IMAGES")
    print("="*60)
    
    # Load embeddings (choose which one to use)
    print("\nLoading embeddings...")
    with open('clip_embeddings.pkl', 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']
    image_paths = data['image_paths']
    model_name = data['model']
    
    print(f"Loaded {len(embeddings)} embeddings from {model_name}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Standardize embeddings
    print("\nStandardizing embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Reduce dimensions
    embeddings_reduced, pca_model = reduce_dimensions(
        embeddings_scaled, 
        n_components=50
    )
    
    # Save reduced embeddings
    with open('embeddings_reduced.pkl', 'wb') as f:
        pickle.dump({
            'embeddings': embeddings_reduced,
            'image_paths': image_paths,
            'scaler': scaler,
            'pca': pca_model,
            'original_model': model_name
        }, f)
    
    # Perform clustering with multiple methods
    results = {}
    
    print("\n" + "="*60)
    print("K-MEANS CLUSTERING")
    print("="*60)
    results['K-means'] = kmeans_clustering(embeddings_reduced, k_values=[3, 5, 7, 10])
    
    print("\n" + "="*60)
    print("HIERARCHICAL CLUSTERING")
    print("="*60)
    results['Hierarchical'] = hierarchical_clustering(
        embeddings_reduced, 
        n_clusters_values=[3, 5, 7, 10]
    )
    
    print("\n" + "="*60)
    print("DBSCAN CLUSTERING")
    print("="*60)
    results['DBSCAN'] = dbscan_clustering(
        embeddings_reduced, 
        eps_values=[1.0, 2.0, 3.0],
        min_samples=3
    )
    
    # Save all clustering results
    with open('clustering_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*60)
    print("✓ CLUSTERING COMPLETE!")
    print("="*60)
    
    # Create summary table
    print("\n" + "="*60)
    print("CLUSTERING COMPARISON TABLE")
    print("="*60)
    
    summary_df = create_results_table(results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('clustering_comparison.csv', index=False)
    print("\n✓ Saved to clustering_comparison.csv")
    
    # Best K-means result
    best_k = 5  # You can choose based on silhouette score
    best_labels = results['K-means'][best_k]['labels']
    
    # Compute inter/intra distances for best result
    print("\n" + "="*60)
    print(f"DETAILED ANALYSIS (K-means, K={best_k})")
    print("="*60)
    
    dist_metrics = compute_inter_intra_distances(embeddings_reduced, best_labels)
    print(f"\nMean Intra-cluster Distance: {dist_metrics['mean_intra_distance']:.3f}")
    print(f"Mean Inter-cluster Distance: {dist_metrics['mean_inter_distance']:.3f}")
    print(f"Separation Ratio: {dist_metrics['separation_ratio']:.3f}")
    
    # Save best clustering results
    np.save('best_cluster_labels.npy', best_labels)
    print("\n✓ Saved best clustering labels to best_cluster_labels.npy")
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("="*60)
    print("  - embeddings_reduced.pkl")
    print("  - clustering_results.pkl")
    print("  - clustering_comparison.csv")
    print("  - best_cluster_labels.npy")
