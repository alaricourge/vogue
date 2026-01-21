"""
Analyse de clustering pour les embeddings visuels
"""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pathlib import Path
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import setup_logger
from ..utils.file_io import load_pickle, save_pickle

logger = setup_logger(__name__)


class ClusterAnalyzer:
    """
    Analyseur de clustering pour les embeddings d'images
    Implémente K-means, Hierarchical et DBSCAN
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise l'analyseur
        
        Args:
            random_state: Seed pour la reproductibilité
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.embeddings_original = None
        self.embeddings_scaled = None
        self.embeddings_reduced = None
        
    def preprocess_embeddings(self, embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Prétraite les embeddings : normalisation et réduction PCA
        
        Args:
            embeddings: Embeddings bruts
            n_components: Nombre de composantes PCA
            
        Returns:
            Embeddings prétraités
        """
        self.embeddings_original = embeddings
        
        # Normalisation
        logger.info(f"Normalisation des embeddings...")
        self.embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Réduction PCA
        logger.info(f"Réduction PCA : {embeddings.shape[1]} -> {n_components} dimensions")
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.embeddings_reduced = self.pca.fit_transform(self.embeddings_scaled)
        
        variance_explained = self.pca.explained_variance_ratio_.sum()
        logger.info(f"Variance expliquée : {variance_explained:.2%}")
        
        return self.embeddings_reduced
    
    def kmeans_clustering(self, embeddings: np.ndarray, k_values: List[int]) -> Dict:
        """
        Clustering K-means pour différentes valeurs de k
        
        Args:
            embeddings: Embeddings à clusteriser
            k_values: Liste des valeurs de k à tester
            
        Returns:
            Dictionnaire des résultats pour chaque k
        """
        logger.info(f"\n{'='*60}")
        logger.info("K-MEANS CLUSTERING")
        logger.info(f"{'='*60}")
        
        results = {}
        
        for k in k_values:
            logger.info(f"\nK-means avec k={k}...")
            
            # Clustering
            kmeans = KMeans(
                n_clusters=k, 
                random_state=self.random_state, 
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(embeddings)
            
            # Métriques
            silhouette = silhouette_score(embeddings, labels)
            db_score = davies_bouldin_score(embeddings, labels)
            ch_score = calinski_harabasz_score(embeddings, labels)
            
            # Distribution des clusters
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            results[k] = {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_,
                'silhouette': silhouette,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score,
                'distribution': distribution,
                'n_clusters': k
            }
            
            logger.info(f"  Silhouette: {silhouette:.3f}")
            logger.info(f"  Davies-Bouldin: {db_score:.3f}")
            logger.info(f"  Calinski-Harabasz: {ch_score:.1f}")
            logger.info(f"  Distribution: {distribution}")
        
        return results
    
    def hierarchical_clustering(self, embeddings: np.ndarray, k_values: List[int]) -> Dict:
        """
        Clustering hiérarchique agglomératif
        
        Args:
            embeddings: Embeddings à clusteriser
            k_values: Liste des valeurs de k à tester
            
        Returns:
            Dictionnaire des résultats pour chaque k
        """
        logger.info(f"\n{'='*60}")
        logger.info("HIERARCHICAL CLUSTERING")
        logger.info(f"{'='*60}")
        
        results = {}
        
        for k in k_values:
            logger.info(f"\nHierarchical avec k={k}...")
            
            # Clustering
            hierarchical = AgglomerativeClustering(n_clusters=k)
            labels = hierarchical.fit_predict(embeddings)
            
            # Métriques
            silhouette = silhouette_score(embeddings, labels)
            db_score = davies_bouldin_score(embeddings, labels)
            ch_score = calinski_harabasz_score(embeddings, labels)
            
            # Distribution
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            results[k] = {
                'labels': labels,
                'silhouette': silhouette,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score,
                'distribution': distribution,
                'n_clusters': k
            }
            
            logger.info(f"  Silhouette: {silhouette:.3f}")
            logger.info(f"  Davies-Bouldin: {db_score:.3f}")
            logger.info(f"  Distribution: {distribution}")
        
        return results
    
    def dbscan_clustering(self, embeddings: np.ndarray, 
                         eps_values: List[float] = [1.0, 2.0, 3.0],
                         min_samples: int = 3) -> Dict:
        """
        Clustering DBSCAN
        
        Args:
            embeddings: Embeddings à clusteriser
            eps_values: Liste des valeurs epsilon à tester
            min_samples: Nombre minimum de samples par cluster
            
        Returns:
            Dictionnaire des résultats pour chaque epsilon
        """
        logger.info(f"\n{'='*60}")
        logger.info("DBSCAN CLUSTERING")
        logger.info(f"{'='*60}")
        
        results = {}
        
        for eps in eps_values:
            logger.info(f"\nDBSCAN avec eps={eps}, min_samples={min_samples}...")
            
            # Clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(embeddings)
            
            # Nombre de clusters (sans le bruit -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Métriques (seulement si au moins 2 clusters)
            if n_clusters >= 2:
                # Retirer les points de bruit pour le calcul des métriques
                mask = labels != -1
                if mask.sum() > 0:
                    silhouette = silhouette_score(embeddings[mask], labels[mask])
                    db_score = davies_bouldin_score(embeddings[mask], labels[mask])
                    ch_score = calinski_harabasz_score(embeddings[mask], labels[mask])
                else:
                    silhouette = db_score = ch_score = None
            else:
                silhouette = db_score = ch_score = None
            
            # Distribution
            unique, counts = np.unique(labels, return_counts=True)
            distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            results[eps] = {
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette,
                'davies_bouldin': db_score,
                'calinski_harabasz': ch_score,
                'distribution': distribution,
                'eps': eps,
                'min_samples': min_samples
            }
            
            logger.info(f"  Clusters trouvés: {n_clusters}")
            logger.info(f"  Points de bruit: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
            if silhouette is not None:
                logger.info(f"  Silhouette: {silhouette:.3f}")
                logger.info(f"  Davies-Bouldin: {db_score:.3f}")
        
        return results
    
    def run_all_methods(self, 
                       embeddings_file: Union[str, Path],
                       k_values: List[int] = [3, 5, 7, 10],
                       output_dir: Union[str, Path] = None,
                       pca_components: int = 50) -> Dict:
        """
        Exécute toutes les méthodes de clustering
        
        Args:
            embeddings_file: Fichier pickle contenant les embeddings
            k_values: Valeurs de k pour K-means et Hierarchical
            output_dir: Dossier de sortie (optionnel)
            pca_components: Nombre de composantes PCA
            
        Returns:
            Dictionnaire complet des résultats
        """
        # Charger les embeddings
        logger.info(f"Chargement des embeddings depuis {embeddings_file}")
        data = load_pickle(embeddings_file)
        embeddings = data['embeddings']
        image_paths = data['image_paths']
        
        logger.info(f"Embeddings chargés : {embeddings.shape}")
        logger.info(f"Images : {len(image_paths)}")
        
        # Prétraitement
        embeddings_reduced = self.preprocess_embeddings(embeddings, n_components=pca_components)
        
        # K-means
        kmeans_results = self.kmeans_clustering(embeddings_reduced, k_values)
        
        # Hierarchical
        hierarchical_results = self.hierarchical_clustering(embeddings_reduced, k_values)
        
        # DBSCAN
        dbscan_results = self.dbscan_clustering(embeddings_reduced)
        
        # Rassembler tous les résultats
        all_results = {
            'kmeans': kmeans_results,
            'hierarchical': hierarchical_results,
            'dbscan': dbscan_results,
            'embeddings_reduced': embeddings_reduced,
            'embeddings_original': embeddings,
            'image_paths': image_paths,
            'pca_variance_explained': self.pca.explained_variance_ratio_.sum(),
            'n_components': pca_components
        }
        
        # Sauvegarder si output_dir est spécifié
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder par méthode
            save_pickle(kmeans_results, output_dir / "kmeans_results.pkl")
            save_pickle(hierarchical_results, output_dir / "hierarchical_results.pkl")
            save_pickle(dbscan_results, output_dir / "dbscan_results.pkl")
            
            # Sauvegarder tout ensemble
            save_pickle(all_results, output_dir / "all_clustering_results.pkl")
            
            logger.info(f"\n✅ Résultats sauvegardés dans {output_dir}")
        
        return all_results
    
    def find_optimal_k(self, results: Dict, method: str = 'silhouette') -> int:
        """
        Trouve le k optimal basé sur une métrique
        
        Args:
            results: Résultats de clustering
            method: Métrique à utiliser ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
            Valeur optimale de k
        """
        k_values = list(results.keys())
        
        if method == 'silhouette':
            # Plus élevé = meilleur
            scores = [results[k]['silhouette'] for k in k_values]
            optimal_k = k_values[np.argmax(scores)]
        elif method == 'davies_bouldin':
            # Plus bas = meilleur
            scores = [results[k]['davies_bouldin'] for k in k_values]
            optimal_k = k_values[np.argmin(scores)]
        elif method == 'calinski_harabasz':
            # Plus élevé = meilleur
            scores = [results[k]['calinski_harabasz'] for k in k_values]
            optimal_k = k_values[np.argmax(scores)]
        else:
            raise ValueError(f"Méthode inconnue: {method}")
        
        logger.info(f"K optimal ({method}): {optimal_k}")
        return optimal_k
