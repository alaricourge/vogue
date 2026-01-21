"""
Script pour visualiser des échantillons d'images de chaque cluster
"""
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
import numpy as np

def visualize_cluster_samples(k=3, n_samples=15, seed=42):
    """
    Visualise des échantillons aléatoires pour chaque cluster
    
    Args:
        k: Nombre de clusters à visualiser
        n_samples: Nombre d'images par cluster
        seed: Seed pour reproductibilité
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"🎨 Visualisation des clusters (k={k})")
    print("=" * 70)
    
    # Chemins
    clustering_file = Path('data/results/clustering/kmeans_results.pkl')
    embeddings_file = Path('data/embeddings/clip_embeddings.pkl')
    output_dir = Path('outputs/figures/cluster_samples')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger les résultats de clustering
    print(f"📂 Chargement des résultats de clustering...")
    with open(clustering_file, 'rb') as f:
        results = pickle.load(f)
    
    if k not in results:
        print(f"❌ Erreur: k={k} non trouvé dans les résultats")
        print(f"   Valeurs disponibles: {list(results.keys())}")
        return
    
    labels = results[k]['labels']
    distribution = results[k]['distribution']
    
    # Charger les chemins des images
    print(f"📂 Chargement des embeddings...")
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    image_paths = data['image_paths']
    
    print(f"\n✅ Données chargées:")
    print(f"   Total images: {len(image_paths)}")
    print(f"   Distribution: {distribution}")
    print()
    
    # Pour chaque cluster
    for cluster_id in range(k):
        print(f"🎨 Cluster {cluster_id}...")
        
        # Trouver toutes les images de ce cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_paths = [image_paths[i] for i in cluster_indices]
        
        print(f"   Images dans ce cluster: {len(cluster_paths)}")
        
        # Échantillonner aléatoirement
        n_to_sample = min(n_samples, len(cluster_paths))
        sampled_paths = random.sample(cluster_paths, n_to_sample)
        
        # Calculer la grille
        n_cols = 5
        n_rows = int(np.ceil(n_to_sample / n_cols))
        
        # Créer la figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        
        # Titre avec statistiques
        fig.suptitle(
            f'Cluster {cluster_id} - {len(cluster_paths)} images ({len(cluster_paths)/len(image_paths)*100:.1f}%)\n'
            f'Échantillon aléatoire de {n_to_sample} images',
            fontsize=16,
            fontweight='bold'
        )
        
        # Aplatir axes si nécessaire
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Afficher les images
        for idx, ax in enumerate(axes):
            ax.axis('off')
            if idx < len(sampled_paths):
                try:
                    img_path = Path(sampled_paths[idx])
                    img = Image.open(img_path)
                    ax.imshow(img)
                    
                    # Ajouter le nom du fichier en petit
                    filename = img_path.name
                    ax.set_title(filename[:30] + '...' if len(filename) > 30 else filename, 
                               fontsize=8, pad=2)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Erreur:\n{str(e)[:50]}', 
                           ha='center', va='center', fontsize=8)
                    print(f"   ⚠️  Erreur avec {sampled_paths[idx]}: {e}")
        
        plt.tight_layout()
        
        # Sauvegarder
        output_path = output_dir / f'cluster_{cluster_id}_k{k}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Sauvegardé: {output_path}")
    
    print(f"\n{'='*70}")
    print(f"✅ Visualisations terminées!")
    print(f"📂 Résultats dans: {output_dir}")
    print(f"{'='*70}")


def create_comparison_grid(k=3, n_samples_per_cluster=5, seed=42):
    """
    Crée une grille de comparaison avec tous les clusters côte à côte
    
    Args:
        k: Nombre de clusters
        n_samples_per_cluster: Nombre d'images par cluster
        seed: Seed pour reproductibilité
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n🔍 Création de la grille de comparaison...")
    
    # Chemins
    clustering_file = Path('data/results/clustering/kmeans_results.pkl')
    embeddings_file = Path('data/embeddings/clip_embeddings.pkl')
    output_dir = Path('outputs/figures/cluster_samples')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger les données
    with open(clustering_file, 'rb') as f:
        results = pickle.load(f)
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    labels = results[k]['labels']
    image_paths = data['image_paths']
    
    # Créer la grille: k clusters x n_samples images
    fig, axes = plt.subplots(k, n_samples_per_cluster, figsize=(3*n_samples_per_cluster, 3*k))
    
    fig.suptitle(f'Comparaison des {k} clusters - {n_samples_per_cluster} échantillons par cluster',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Pour chaque cluster
    for cluster_id in range(k):
        # Trouver les images
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_paths = [image_paths[i] for i in cluster_indices]
        
        # Échantillonner
        n_to_sample = min(n_samples_per_cluster, len(cluster_paths))
        sampled_paths = random.sample(cluster_paths, n_to_sample)
        
        # Afficher
        for col_idx in range(n_samples_per_cluster):
            ax = axes[cluster_id, col_idx] if k > 1 else axes[col_idx]
            ax.axis('off')
            
            # Label du cluster sur la première colonne
            if col_idx == 0:
                ax.text(-0.1, 0.5, f'Cluster {cluster_id}\n({len(cluster_paths)} imgs)',
                       transform=ax.transAxes,
                       fontsize=12, fontweight='bold',
                       ha='right', va='center')
            
            if col_idx < len(sampled_paths):
                try:
                    img = Image.open(sampled_paths[col_idx])
                    ax.imshow(img)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Erreur', ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = output_dir / f'comparison_grid_k{k}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Grille de comparaison sauvegardée: {output_path}")


def analyze_cluster_metadata(k=3):
    """
    Analyse les métadonnées extraites des noms de fichiers
    """
    print(f"\n📊 Analyse des métadonnées (k={k})...")
    
    # Chemins
    clustering_file = Path('data/results/clustering/kmeans_results.pkl')
    embeddings_file = Path('data/embeddings/clip_embeddings.pkl')
    
    # Charger
    with open(clustering_file, 'rb') as f:
        results = pickle.load(f)
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    labels = results[k]['labels']
    image_paths = data['image_paths']
    
    print(f"\n{'='*70}")
    
    # Pour chaque cluster
    for cluster_id in range(k):
        print(f"\n🎨 CLUSTER {cluster_id}")
        print("-" * 70)
        
        # Trouver les images
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_paths = [Path(image_paths[i]).name for i in cluster_indices]
        
        # Extraire designers (partie avant le premier _)
        designers = {}
        saisons = {}
        annees = {}
        
        for filename in cluster_paths:
            parts = filename.replace('.jpg', '').replace('.png', '').split('_')
            
            if len(parts) >= 3:
                designer = parts[0]
                saison = parts[1] if len(parts) > 1 else 'unknown'
                annee = parts[2] if len(parts) > 2 else 'unknown'
                
                designers[designer] = designers.get(designer, 0) + 1
                saisons[saison] = saisons.get(saison, 0) + 1
                annees[annee] = annees.get(annee, 0) + 1
        
        # Top 5 designers
        print(f"\n👔 Top 5 Designers:")
        for designer, count in sorted(designers.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {designer}: {count} images ({count/len(cluster_paths)*100:.1f}%)")
        
        # Top 5 saisons
        print(f"\n📅 Top 5 Saisons:")
        for saison, count in sorted(saisons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {saison}: {count} images ({count/len(cluster_paths)*100:.1f}%)")
        
        # Top 5 années
        print(f"\n📆 Top 5 Années:")
        for annee, count in sorted(annees.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {annee}: {count} images ({count/len(cluster_paths)*100:.1f}%)")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    print("🎨 Fashion Runway Analysis - Visualisation des Clusters")
    print("=" * 70)
    print()
    
    # 1. Visualiser chaque cluster séparément (15 images)
    visualize_cluster_samples(k=3, n_samples=15, seed=42)
    
    # 2. Créer une grille de comparaison
    create_comparison_grid(k=3, n_samples_per_cluster=5, seed=42)
    
    # 3. Analyser les métadonnées
    analyze_cluster_metadata(k=3)
    
    print("\n✨ Terminé! Ouvrez le dossier outputs/figures/cluster_samples/")
