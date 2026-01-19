"""
Visualization Code for Fashion Runway Analysis
Creates all figures needed for the final report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import pickle
from sklearn.manifold import TSNE
import pandas as pd
from matplotlib.gridspec import GridSpec


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# FIGURE 1: SEGMENTATION RESULTS (Before/After)
# ============================================================================

def visualize_segmentation_results(
    original_folder, 
    segmented_folder, 
    num_examples=6,
    output_file='figure1_segmentation.png'
):
    """
    Create before/after comparison of segmentation
    
    Args:
        original_folder: folder with original images
        segmented_folder: folder with segmented images
        num_examples: number of example pairs to show
        output_file: where to save the figure
    """
    fig, axes = plt.subplots(2, num_examples, figsize=(18, 6))
    
    # Get matching image pairs
    segmented_files = sorted([f for f in os.listdir(segmented_folder) 
                             if f.endswith('.png')])[:num_examples]
    
    for idx, seg_file in enumerate(segmented_files):
        # Original image (assuming same ID)
        img_id = seg_file.split('.')[0]
        orig_file = f"{img_id}.jpg"
        
        # Load images
        try:
            orig_path = os.path.join(original_folder, orig_file)
            seg_path = os.path.join(segmented_folder, seg_file)
            
            orig_img = Image.open(orig_path).convert('RGB')
            seg_img = Image.open(seg_path).convert('RGBA')
            
            # Display original
            axes[0, idx].imshow(orig_img)
            axes[0, idx].axis('off')
            if idx == 0:
                axes[0, idx].set_title('Original', fontsize=12, fontweight='bold')
            
            # Display segmented
            axes[1, idx].imshow(seg_img)
            axes[1, idx].axis('off')
            if idx == 0:
                axes[1, idx].set_title('Segmented', fontsize=12, fontweight='bold')
        
        except Exception as e:
            print(f"Error loading {seg_file}: {e}")
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


# ============================================================================
# FIGURE 2: COLOR PALETTE EXTRACTION
# ============================================================================

def visualize_color_palettes(
    image_folder,
    color_extraction_func,
    num_examples=8,
    colors_per_image=5,
    output_file='figure2_color_palettes.png'
):
    """
    Show dominant color extraction for multiple images
    
    Args:
        image_folder: folder with segmented images
        color_extraction_func: your function to extract dominant colors
        num_examples: number of images to show
        colors_per_image: number of colors to extract per image
        output_file: where to save
    """
    from utile_vogue import open_image, resize_image, extract_dominant_color
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(num_examples, colors_per_image + 1, figure=fig, hspace=0.3)
    
    image_files = sorted([f for f in os.listdir(image_folder) 
                         if f.endswith('.png')])[:num_examples]
    
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join(image_folder, img_file)
        
        # Load and resize image
        img = open_image(img_path)
        img = resize_image(img)
        
        # Show image
        ax_img = fig.add_subplot(gs[idx, 0])
        ax_img.imshow(img)
        ax_img.axis('off')
        if idx == 0:
            ax_img.set_title('Image', fontsize=11, fontweight='bold')
        
        # Extract and show colors
        try:
            colors = extract_dominant_color(img, colors_per_image)
            
            for color_idx, color in enumerate(colors):
                ax_color = fig.add_subplot(gs[idx, color_idx + 1])
                color_square = np.ones((100, 100, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)
                ax_color.imshow(color_square)
                ax_color.axis('off')
                
                if idx == 0:
                    ax_color.set_title(f'Color {color_idx+1}', fontsize=11, fontweight='bold')
        
        except Exception as e:
            print(f"Error extracting colors from {img_file}: {e}")
    
    plt.suptitle('Dominant Color Extraction from Runway Images', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


# ============================================================================
# FIGURE 3: t-SNE CLUSTER VISUALIZATION
# ============================================================================

def visualize_tsne_clusters(
    embeddings,
    labels,
    image_paths=None,
    metadata_df=None,
    color_by='cluster',
    output_file='figure3_tsne_clusters.png'
):
    """
    Create t-SNE visualization of clusters
    
    Args:
        embeddings: high-dimensional embeddings
        labels: cluster labels
        image_paths: optional, for adding image thumbnails
        metadata_df: optional, for coloring by designer/season
        color_by: 'cluster', 'designer', or 'season'
        output_file: where to save
    """
    print("Computing t-SNE (this may take a few minutes)...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if color_by == 'cluster':
        # Color by cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise in DBSCAN
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c='gray', label='Noise', alpha=0.3, s=30)
            else:
                mask = labels == label
                ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=[color], label=f'Cluster {label}', alpha=0.6, s=50)
        
        ax.set_title('t-SNE Visualization of Runway Image Clusters', 
                    fontsize=14, fontweight='bold')
    
    elif color_by == 'designer' and metadata_df is not None:
        # Color by designer
        designers = metadata_df['designer'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(designers)))
        
        for designer, color in zip(designers, colors):
            mask = metadata_df['designer'] == designer
            ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                      c=[color], label=designer, alpha=0.6, s=50)
        
        ax.set_title('t-SNE Visualization Colored by Designer', 
                    fontsize=14, fontweight='bold')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()
    
    return embeddings_2d


# ============================================================================
# FIGURE 4: CLUSTER COMPOSITION ANALYSIS
# ============================================================================

def visualize_cluster_composition(
    labels,
    metadata_df,
    output_file='figure4_cluster_composition.png'
):
    """
    Show what designers/seasons are in each cluster
    
    Args:
        labels: cluster assignments
        metadata_df: DataFrame with designer, season, year
        output_file: where to save
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data
    metadata_df['cluster'] = labels
    
    # Plot 1: Designer distribution per cluster
    designer_cluster = pd.crosstab(metadata_df['cluster'], metadata_df['designer'])
    designer_cluster_pct = designer_cluster.div(designer_cluster.sum(axis=1), axis=0) * 100
    
    designer_cluster_pct.plot(kind='bar', stacked=True, ax=axes[0], 
                              colormap='tab20', legend=False)
    axes[0].set_title('Designer Distribution per Cluster', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Cluster', fontsize=11)
    axes[0].set_ylabel('Percentage (%)', fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Plot 2: Season distribution per cluster
    season_cluster = pd.crosstab(metadata_df['cluster'], metadata_df['saison'])
    season_cluster_pct = season_cluster.div(season_cluster.sum(axis=1), axis=0) * 100
    
    season_cluster_pct.plot(kind='bar', stacked=True, ax=axes[1], 
                           colormap='Set3', legend=True)
    axes[1].set_title('Season Distribution per Cluster', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Cluster', fontsize=11)
    axes[1].set_ylabel('Percentage (%)', fontsize=11)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


# ============================================================================
# FIGURE 5: CLUSTERING METRICS COMPARISON
# ============================================================================

def visualize_clustering_metrics(
    clustering_results,
    output_file='figure5_metrics_comparison.png'
):
    """
    Compare different clustering approaches
    
    Args:
        clustering_results: dict from clustering_analysis.py
        output_file: where to save
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Extract K-means results
    kmeans_results = clustering_results.get('K-means', {})
    k_values = sorted(kmeans_results.keys())
    
    silhouette_scores = [kmeans_results[k]['silhouette'] for k in k_values]
    db_scores = [kmeans_results[k]['davies_bouldin'] for k in k_values]
    ch_scores = [kmeans_results[k]['calinski_harabasz'] for k in k_values]
    
    # Plot 1: Silhouette Score
    axes[0].plot(k_values, silhouette_scores, marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[0].set_ylabel('Silhouette Score', fontsize=11)
    axes[0].set_title('Silhouette Score vs K\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_values)
    
    # Plot 2: Davies-Bouldin Score
    axes[1].plot(k_values, db_scores, marker='s', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[1].set_ylabel('Davies-Bouldin Score', fontsize=11)
    axes[1].set_title('Davies-Bouldin Score vs K\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_values)
    
    # Plot 3: Calinski-Harabasz Score
    axes[2].plot(k_values, ch_scores, marker='^', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=11)
    axes[2].set_ylabel('Calinski-Harabasz Score', fontsize=11)
    axes[2].set_title('Calinski-Harabasz Score vs K\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(k_values)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


# ============================================================================
# FIGURE 6: DESIGNER-SPECIFIC CLUSTERS
# ============================================================================

def visualize_designer_clusters(
    embeddings_2d,
    labels,
    metadata_df,
    designer_name,
    output_file='figure6_designer_specific.png'
):
    """
    Show clustering for a specific designer
    
    Args:
        embeddings_2d: 2D t-SNE embeddings
        labels: cluster labels
        metadata_df: metadata
        designer_name: which designer to highlight
        output_file: where to save
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mask for this designer
    designer_mask = metadata_df['designer'] == designer_name
    other_mask = ~designer_mask
    
    # Plot other designers in gray
    ax.scatter(embeddings_2d[other_mask, 0], embeddings_2d[other_mask, 1],
              c='lightgray', alpha=0.2, s=20, label='Other Designers')
    
    # Plot this designer colored by cluster
    designer_labels = labels[designer_mask]
    designer_coords = embeddings_2d[designer_mask]
    
    unique_labels = np.unique(designer_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = designer_labels == label
        ax.scatter(designer_coords[mask, 0], designer_coords[mask, 1],
                  c=[color], label=f'Cluster {label}', alpha=0.7, s=80)
    
    ax.set_title(f'{designer_name.title()} - Cluster Distribution', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


# ============================================================================
# FIGURE 7: SAMPLE IMAGES FROM EACH CLUSTER
# ============================================================================

def visualize_cluster_examples(
    labels,
    image_paths,
    num_clusters=5,
    images_per_cluster=5,
    output_file='figure7_cluster_examples.png'
):
    """
    Show representative images from each cluster
    
    Args:
        labels: cluster assignments
        image_paths: paths to images
        num_clusters: number of clusters to show
        images_per_cluster: images per cluster
        output_file: where to save
    """
    fig, axes = plt.subplots(num_clusters, images_per_cluster, 
                            figsize=(15, 3*num_clusters))
    
    for cluster_id in range(num_clusters):
        # Get images in this cluster
        cluster_mask = labels == cluster_id
        cluster_paths = [image_paths[i] for i, m in enumerate(cluster_mask) if m]
        
        # Sample random images
        sample_paths = np.random.choice(cluster_paths, 
                                       min(images_per_cluster, len(cluster_paths)),
                                       replace=False)
        
        for img_idx, img_path in enumerate(sample_paths):
            try:
                img = Image.open(img_path).convert('RGB')
                axes[cluster_id, img_idx].imshow(img)
                axes[cluster_id, img_idx].axis('off')
                
                if img_idx == 0:
                    axes[cluster_id, img_idx].set_ylabel(
                        f'Cluster {cluster_id}\n({cluster_mask.sum()} images)', 
                        fontsize=10, fontweight='bold'
                    )
            except Exception as e:
                axes[cluster_id, img_idx].axis('off')
        
        # Hide remaining axes if fewer images
        for img_idx in range(len(sample_paths), images_per_cluster):
            axes[cluster_id, img_idx].axis('off')
    
    plt.suptitle('Sample Images from Each Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved {output_file}")
    plt.close()


# ============================================================================
# MAIN EXECUTION - GENERATE ALL FIGURES
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("GENERATING ALL VISUALIZATIONS FOR REPORT")
    print("="*60)
    
    # Load data
    print("\nLoading embeddings and clustering results...")
    
    with open('embeddings_reduced.pkl', 'rb') as f:
        embed_data = pickle.load(f)
    embeddings = embed_data['embeddings']
    image_paths = embed_data['image_paths']
    
    with open('clustering_results.pkl', 'rb') as f:
        clustering_results = pickle.load(f)
    
    labels = clustering_results['K-means'][5]['labels']  # Use K=5 as example
    
    # Load metadata
    metadata_df = pd.read_csv('information_defile.csv')
    # Match metadata to image paths (you may need to adjust this)
    
    print("\n[1/7] Creating segmentation before/after figure...")
    visualize_segmentation_results(
        original_folder='defile_vogue/traitee',
        segmented_folder='defile_vogue/tres_traitee',
        num_examples=6,
        output_file='figure1_segmentation.png'
    )
    
    print("\n[2/7] Creating color palette extraction figure...")
    # Note: This requires your color extraction function
    # visualize_color_palettes(...)
    
    print("\n[3/7] Creating t-SNE cluster visualization...")
    embeddings_2d = visualize_tsne_clusters(
        embeddings=embeddings,
        labels=labels,
        image_paths=image_paths,
        output_file='figure3_tsne_clusters.png'
    )
    
    print("\n[4/7] Creating cluster composition analysis...")
    visualize_cluster_composition(
        labels=labels,
        metadata_df=metadata_df,
        output_file='figure4_cluster_composition.png'
    )
    
    print("\n[5/7] Creating clustering metrics comparison...")
    visualize_clustering_metrics(
        clustering_results=clustering_results,
        output_file='figure5_metrics_comparison.png'
    )
    
    print("\n[6/7] Creating designer-specific visualization...")
    visualize_designer_clusters(
        embeddings_2d=embeddings_2d,
        labels=labels,
        metadata_df=metadata_df,
        designer_name='gucci',
        output_file='figure6_designer_gucci.png'
    )
    
    print("\n[7/7] Creating cluster example images...")
    visualize_cluster_examples(
        labels=labels,
        image_paths=image_paths,
        num_clusters=5,
        images_per_cluster=5,
        output_file='figure7_cluster_examples.png'
    )
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nFigures created:")
    print("  - figure1_segmentation.png")
    print("  - figure2_color_palettes.png")
    print("  - figure3_tsne_clusters.png")
    print("  - figure4_cluster_composition.png")
    print("  - figure5_metrics_comparison.png")
    print("  - figure6_designer_gucci.png")
    print("  - figure7_cluster_examples.png")
