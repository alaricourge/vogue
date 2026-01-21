"""
Script bonus : Analyse des couleurs dominantes par cluster
"""
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans as ColorKMeans


def extract_dominant_colors(image_path, n_colors=5):
    """
    Extrait les couleurs dominantes d'une image
    
    Args:
        image_path: Chemin de l'image
        n_colors: Nombre de couleurs à extraire
        
    Returns:
        Liste de couleurs RGB
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((150, 150))  # Réduire pour accélérer
        
        # Convertir en array
        pixels = np.array(img).reshape(-1, 3)
        
        # Clustering des couleurs
        kmeans = ColorKMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return colors
        
    except Exception as e:
        return None


def analyze_cluster_colors(k=3, n_images_sample=50, n_colors=5):
    """
    Analyse les couleurs dominantes pour chaque cluster
    
    Args:
        k: Nombre de clusters à analyser
        n_images_sample: Nombre d'images à échantillonner par cluster
        n_colors: Nombre de couleurs dominantes à extraire par image
    """
    print(f"🎨 Analyse des couleurs par cluster (k={k})")
    print("=" * 70)
    
    # Chemins
    clustering_file = Path('data/results/clustering/kmeans_results.pkl')
    embeddings_file = Path('data/embeddings/clip_embeddings.pkl')
    output_dir = Path('outputs/figures/cluster_colors')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger les données
    print("📂 Chargement des données...")
    with open(clustering_file, 'rb') as f:
        results = pickle.load(f)
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    labels = results[k]['labels']
    image_paths = data['image_paths']
    
    # Analyser chaque cluster
    cluster_palettes = {}
    
    for cluster_id in range(k):
        print(f"\n🎨 Analyse du Cluster {cluster_id}...")
        
        # Trouver les images
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_paths = [image_paths[i] for i in cluster_indices]
        
        # Échantillonner
        import random
        n_sample = min(n_images_sample, len(cluster_paths))
        sampled_paths = random.sample(cluster_paths, n_sample)
        
        print(f"   Extraction des couleurs de {n_sample} images...")
        
        # Extraire les couleurs de toutes les images échantillonnées
        all_colors = []
        for img_path in sampled_paths:
            colors = extract_dominant_colors(img_path, n_colors=n_colors)
            if colors is not None:
                all_colors.extend(colors)
        
        if not all_colors:
            print(f"   ⚠️  Aucune couleur extraite pour le cluster {cluster_id}")
            continue
        
        # Re-clusteriser toutes les couleurs pour trouver la palette du cluster
        all_colors = np.array(all_colors)
        kmeans = ColorKMeans(n_clusters=10, random_state=42, n_init=10)
        kmeans.fit(all_colors)
        
        palette = kmeans.cluster_centers_.astype(int)
        cluster_palettes[cluster_id] = palette
        
        print(f"   ✅ Palette de {len(palette)} couleurs extraite")
    
    # Visualiser les palettes
    print(f"\n📊 Création des visualisations...")
    
    fig, axes = plt.subplots(k, 1, figsize=(12, 2*k))
    if k == 1:
        axes = [axes]
    
    fig.suptitle(f'Palettes de couleurs dominantes par cluster (k={k})',
                fontsize=16, fontweight='bold')
    
    for cluster_id in range(k):
        ax = axes[cluster_id]
        
        if cluster_id in cluster_palettes:
            palette = cluster_palettes[cluster_id]
            
            # Créer une image de la palette
            palette_img = np.zeros((100, len(palette) * 100, 3), dtype=np.uint8)
            for i, color in enumerate(palette):
                palette_img[:, i*100:(i+1)*100] = color
            
            ax.imshow(palette_img)
            ax.axis('off')
            ax.set_title(f'Cluster {cluster_id}', fontsize=12, fontweight='bold', pad=10)
            
            # Ajouter les valeurs RGB sous chaque couleur
            for i, color in enumerate(palette):
                ax.text(i*100 + 50, 110, f'RGB\n({color[0]},{color[1]},{color[2]})',
                       ha='center', va='top', fontsize=8)
        else:
            ax.text(0.5, 0.5, f'Cluster {cluster_id}\n(pas de données)',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    output_path = output_dir / f'color_palettes_k{k}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Palette sauvegardée: {output_path}")
    
    # Créer une visualisation individuelle pour chaque cluster
    for cluster_id in range(k):
        if cluster_id not in cluster_palettes:
            continue
        
        palette = cluster_palettes[cluster_id]
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 3))
        
        # Créer l'image de palette
        palette_img = np.zeros((200, len(palette) * 150, 3), dtype=np.uint8)
        for i, color in enumerate(palette):
            palette_img[:, i*150:(i+1)*150] = color
        
        ax.imshow(palette_img)
        ax.axis('off')
        ax.set_title(f'Cluster {cluster_id} - Palette de couleurs dominantes',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Ajouter les valeurs RGB et HEX
        for i, color in enumerate(palette):
            hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            ax.text(i*150 + 75, 220, 
                   f'RGB: ({color[0]},{color[1]},{color[2]})\nHEX: {hex_color}',
                   ha='center', va='top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = output_dir / f'cluster_{cluster_id}_palette.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"\n{'='*70}")
    print(f"✅ Analyse des couleurs terminée!")
    print(f"📂 Résultats dans: {output_dir}")
    print(f"{'='*70}")


def create_color_distribution_chart(k=3):
    """
    Crée un graphique montrant la distribution de luminosité par cluster
    """
    print(f"\n📊 Création du graphique de distribution des couleurs...")
    
    # Chemins
    clustering_file = Path('data/results/clustering/kmeans_results.pkl')
    embeddings_file = Path('data/embeddings/clip_embeddings.pkl')
    output_dir = Path('outputs/figures/cluster_colors')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger
    with open(clustering_file, 'rb') as f:
        results = pickle.load(f)
    
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)
    
    labels = results[k]['labels']
    image_paths = data['image_paths']
    
    # Analyser la luminosité moyenne par cluster
    cluster_brightness = {i: [] for i in range(k)}
    
    import random
    for cluster_id in range(k):
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_paths = [image_paths[i] for i in cluster_indices]
        
        # Échantillonner 100 images
        sampled = random.sample(cluster_paths, min(100, len(cluster_paths)))
        
        for img_path in sampled:
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                brightness = np.mean(img_array)
                cluster_brightness[cluster_id].append(brightness)
            except:
                pass
    
    # Créer le graphique
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for cluster_id in range(k):
        brightness = cluster_brightness[cluster_id]
        if brightness:
            ax.hist(brightness, bins=30, alpha=0.6, label=f'Cluster {cluster_id}')
    
    ax.set_xlabel('Luminosité moyenne', fontsize=12)
    ax.set_ylabel('Nombre d\'images', fontsize=12)
    ax.set_title('Distribution de la luminosité par cluster', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'brightness_distribution_k{k}.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graphique de distribution sauvegardé: {output_path}")


if __name__ == "__main__":
    print("🎨 Analyse des couleurs dominantes par cluster")
    print("=" * 70)
    print()
    
    # Analyser les couleurs
    analyze_cluster_colors(k=3, n_images_sample=50, n_colors=5)
    
    # Distribution de luminosité
    create_color_distribution_chart(k=3)
    
    print("\n✨ Terminé! Ouvrez le dossier outputs/figures/cluster_colors/")
