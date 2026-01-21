"""
Visualisations des résultats de clustering et d'analyse
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..utils.logger import setup_logger
from ..utils.file_io import load_pickle

logger = setup_logger(__name__)


class ResultsVisualizer:
    """
    Génère les visualisations pour les résultats de clustering
    """
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'husl', dpi: int = 300):
        """
        Initialise le visualiseur
        
        Args:
            style: Style seaborn
            palette: Palette de couleurs
            dpi: Résolution des figures
        """
        sns.set_style(style)
        sns.set_palette(palette)
        self.dpi = dpi
        
    def plot_clustering_metrics(self, results: Dict, output_path: Union[str, Path]):
        """
        Graphique des métriques de clustering
        
        Args:
            results: Résultats de K-means ou Hierarchical
            output_path: Chemin de sortie
        """
        k_values = sorted(results.keys())
        
        # Extraire les métriques
        silhouettes = [results[k]['silhouette'] for k in k_values]
        db_scores = [results[k]['davies_bouldin'] for k in k_values]
        ch_scores = [results[k]['calinski_harabasz'] for k in k_values]
        
        # Créer la figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Silhouette Score
        axes[0].plot(k_values, silhouettes, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Nombre de clusters (k)', fontsize=11)
        axes[0].set_ylabel('Silhouette Score', fontsize=11)
        axes[0].set_title('Silhouette Score\n(plus élevé = meilleur)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(k_values)
        
        # Marquer le maximum
        max_idx = np.argmax(silhouettes)
        axes[0].scatter(k_values[max_idx], silhouettes[max_idx], 
                       color='red', s=200, zorder=5, alpha=0.5)
        axes[0].annotate(f'Max: k={k_values[max_idx]}', 
                        xy=(k_values[max_idx], silhouettes[max_idx]),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold')
        
        # Davies-Bouldin Score
        axes[1].plot(k_values, db_scores, marker='o', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Nombre de clusters (k)', fontsize=11)
        axes[1].set_ylabel('Davies-Bouldin Score', fontsize=11)
        axes[1].set_title('Davies-Bouldin Score\n(plus bas = meilleur)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(k_values)
        
        # Marquer le minimum
        min_idx = np.argmin(db_scores)
        axes[1].scatter(k_values[min_idx], db_scores[min_idx], 
                       color='red', s=200, zorder=5, alpha=0.5)
        axes[1].annotate(f'Min: k={k_values[min_idx]}', 
                        xy=(k_values[min_idx], db_scores[min_idx]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold')
        
        # Calinski-Harabasz Score
        axes[2].plot(k_values, ch_scores, marker='o', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Nombre de clusters (k)', fontsize=11)
        axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=11)
        axes[2].set_title('Calinski-Harabasz Score\n(plus élevé = meilleur)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(k_values)
        
        # Marquer le maximum
        max_idx = np.argmax(ch_scores)
        axes[2].scatter(k_values[max_idx], ch_scores[max_idx], 
                       color='red', s=200, zorder=5, alpha=0.5)
        axes[2].annotate(f'Max: k={k_values[max_idx]}', 
                        xy=(k_values[max_idx], ch_scores[max_idx]),
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=9, color='red', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Graphique des métriques sauvegardé : {output_path}")
    
    def plot_cluster_distribution(self, results: Dict, k: int, output_path: Union[str, Path]):
        """
        Graphique de la distribution des clusters pour un k donné
        
        Args:
            results: Résultats de clustering
            k: Valeur de k
            output_path: Chemin de sortie
        """
        distribution = results[k]['distribution']
        
        # Trier par cluster ID
        clusters = sorted(distribution.keys())
        counts = [distribution[c] for c in clusters]
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(clusters, counts, color=sns.color_palette("husl", len(clusters)))
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Nombre d\'images', fontsize=12)
        ax.set_title(f'Distribution des images par cluster (k={k})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Graphique de distribution sauvegardé : {output_path}")
    
    def plot_inertia_elbow(self, results: Dict, output_path: Union[str, Path]):
        """
        Graphique du coude (elbow method) pour K-means
        
        Args:
            results: Résultats de K-means
            output_path: Chemin de sortie
        """
        k_values = sorted(results.keys())
        inertias = [results[k]['inertia'] for k in k_values]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(k_values, inertias, marker='o', linewidth=2, markersize=10)
        ax.set_xlabel('Nombre de clusters (k)', fontsize=12)
        ax.set_ylabel('Inertie (somme des distances carrées)', fontsize=12)
        ax.set_title('Méthode du Coude (Elbow Method)\nPour déterminer le k optimal', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Graphique du coude sauvegardé : {output_path}")
    
    def plot_comparison_methods(self, kmeans_results: Dict, hierarchical_results: Dict,
                               output_path: Union[str, Path]):
        """
        Compare K-means et Hierarchical côte à côte
        
        Args:
            kmeans_results: Résultats K-means
            hierarchical_results: Résultats Hierarchical
            output_path: Chemin de sortie
        """
        k_values = sorted(kmeans_results.keys())
        
        # Extraire les silhouettes
        kmeans_silhouettes = [kmeans_results[k]['silhouette'] for k in k_values]
        hier_silhouettes = [hierarchical_results[k]['silhouette'] for k in k_values]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(k_values))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, kmeans_silhouettes, width, 
                      label='K-means', color='steelblue')
        bars2 = ax.bar(x + width/2, hier_silhouettes, width, 
                      label='Hierarchical', color='coral')
        
        ax.set_xlabel('Nombre de clusters (k)', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Comparaison K-means vs Hierarchical\n(Silhouette Score)', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(k_values)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Graphique de comparaison sauvegardé : {output_path}")
    
    def plot_image_grid(self, image_paths: List[str], labels: np.ndarray, 
                       cluster_id: int, output_path: Union[str, Path],
                       max_images: int = 25):
        """
        Grille d'images pour un cluster donné
        
        Args:
            image_paths: Chemins des images
            labels: Labels de clustering
            cluster_id: ID du cluster à visualiser
            output_path: Chemin de sortie
            max_images: Nombre maximum d'images à afficher
        """
        # Trouver les images du cluster
        cluster_mask = labels == cluster_id
        cluster_paths = [path for path, mask in zip(image_paths, cluster_mask) if mask]
        
        # Limiter le nombre d'images
        n_images = min(len(cluster_paths), max_images)
        cluster_paths = cluster_paths[:n_images]
        
        # Calculer la grille
        grid_size = int(np.ceil(np.sqrt(n_images)))
        
        # Créer la figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for idx, ax in enumerate(axes):
            ax.axis('off')
            if idx < n_images:
                try:
                    img = Image.open(cluster_paths[idx])
                    ax.imshow(img)
                except Exception as e:
                    logger.warning(f"Erreur chargement image {cluster_paths[idx]}: {e}")
        
        plt.suptitle(f'Cluster {cluster_id} - {len(cluster_paths)} images', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')  # Réduire DPI pour grilles
        plt.close()
        
        logger.info(f"✓ Grille d'images cluster {cluster_id} sauvegardée : {output_path}")
    
    def generate_all_figures(self, 
                           clustering_results_file: Union[str, Path],
                           output_dir: Union[str, Path],
                           generate_grids: bool = False):
        """
        Génère toutes les visualisations
        
        Args:
            clustering_results_file: Fichier des résultats de clustering
            output_dir: Dossier de sortie
            generate_grids: Générer les grilles d'images (peut être long)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info("GÉNÉRATION DES VISUALISATIONS")
        logger.info(f"{'='*60}")
        
        # Charger les résultats
        logger.info(f"Chargement des résultats depuis {clustering_results_file}")
        
        # Déterminer quel fichier on a reçu
        results = load_pickle(clustering_results_file)
        
        # Si c'est le fichier all_clustering_results.pkl
        if 'kmeans' in results and 'hierarchical' in results:
            kmeans_results = results['kmeans']
            hierarchical_results = results['hierarchical']
            image_paths = results.get('image_paths', [])
        # Si c'est juste kmeans_results.pkl
        else:
            kmeans_results = results
            hierarchical_results = None
            image_paths = []
        
        # 1. Métriques K-means
        self.plot_clustering_metrics(
            kmeans_results, 
            output_dir / 'kmeans_metrics.png'
        )
        
        # 2. Elbow method
        self.plot_inertia_elbow(
            kmeans_results,
            output_dir / 'kmeans_elbow.png'
        )
        
        # 3. Distribution pour différents k
        for k in [3, 5, 7, 10]:
            if k in kmeans_results:
                self.plot_cluster_distribution(
                    kmeans_results, 
                    k,
                    output_dir / f'distribution_k{k}.png'
                )
        
        # 4. Comparaison si on a hierarchical
        if hierarchical_results:
            self.plot_clustering_metrics(
                hierarchical_results,
                output_dir / 'hierarchical_metrics.png'
            )
            
            self.plot_comparison_methods(
                kmeans_results,
                hierarchical_results,
                output_dir / 'comparison_kmeans_vs_hierarchical.png'
            )
        
        # 5. Grilles d'images (optionnel)
        if generate_grids and image_paths:
            logger.info("\nGénération des grilles d'images...")
            
            # Utiliser k=5 comme exemple
            k = 5
            if k in kmeans_results:
                labels = kmeans_results[k]['labels']
                
                grids_dir = output_dir / 'image_grids'
                grids_dir.mkdir(exist_ok=True)
                
                for cluster_id in range(k):
                    self.plot_image_grid(
                        image_paths,
                        labels,
                        cluster_id,
                        grids_dir / f'cluster_{cluster_id}.png',
                        max_images=25
                    )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Toutes les visualisations sauvegardées dans {output_dir}")
        logger.info(f"{'='*60}")
    
    def plot_pca_variance(self, pca, output_path: Union[str, Path]):
        """
        Graphique de la variance expliquée par PCA
        
        Args:
            pca: Objet PCA fitté
            output_path: Chemin de sortie
        """
        variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Variance par composante
        ax1.bar(range(1, len(variance_ratio) + 1), variance_ratio)
        ax1.set_xlabel('Composante PCA', fontsize=11)
        ax1.set_ylabel('Ratio de variance expliquée', fontsize=11)
        ax1.set_title('Variance Expliquée par Composante', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Variance cumulée
        ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                marker='o', linewidth=2)
        ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
        ax2.set_xlabel('Nombre de composantes', fontsize=11)
        ax2.set_ylabel('Variance cumulée', fontsize=11)
        ax2.set_title('Variance Cumulée', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Graphique PCA sauvegardé : {output_path}")
