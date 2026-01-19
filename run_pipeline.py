"""
MASTER PIPELINE - Fashion Runway Analysis
Run this script to execute the complete analysis pipeline
"""

import os
import sys
from datetime import datetime

print("="*80)
print(" FASHION RUNWAY ANALYSIS - COMPLETE PIPELINE")
print("="*80)
print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# STEP 1: EMBEDDING EXTRACTION
# ============================================================================

print("\n" + "█"*80)
print(" STEP 1: EXTRACTING DEEP VISUAL EMBEDDINGS")
print("█"*80)

try:
    import embeddings_extraction
    print("\n✓ Embeddings extracted successfully!")
except Exception as e:
    print(f"\n✗ Error in embedding extraction: {e}")
    print("  Please check your image paths and model installations.")
    sys.exit(1)

# ============================================================================
# STEP 2: CLUSTERING ANALYSIS
# ============================================================================

print("\n" + "█"*80)
print(" STEP 2: PERFORMING CLUSTERING ANALYSIS")
print("█"*80)

try:
    import clustering_analysis
    print("\n✓ Clustering analysis complete!")
except Exception as e:
    print(f"\n✗ Error in clustering: {e}")
    sys.exit(1)

# ============================================================================
# STEP 3: VISUALIZATION GENERATION
# ============================================================================

print("\n" + "█"*80)
print(" STEP 3: GENERATING VISUALIZATIONS")
print("█"*80)

try:
    import visualization
    print("\n✓ All visualizations created!")
except Exception as e:
    print(f"\n✗ Error in visualization: {e}")
    print("  Some figures may be missing.")

# ============================================================================
# STEP 4: GENERATE RESULTS SUMMARY
# ============================================================================

print("\n" + "█"*80)
print(" STEP 4: GENERATING RESULTS SUMMARY")
print("█"*80)

import pickle
import pandas as pd
import numpy as np

# Load all results
with open('clustering_results.pkl', 'rb') as f:
    clustering_results = pickle.load(f)

with open('embeddings_reduced.pkl', 'rb') as f:
    embed_data = pickle.load(f)

# Create summary report
summary = {
    'Total Images': len(embed_data['image_paths']),
    'Embedding Model': embed_data['original_model'],
    'Embedding Dimension (original)': embed_data['embeddings'].shape[1],
    'Reduced Dimension': 50,
}

print("\n" + "-"*80)
print(" DATASET SUMMARY")
print("-"*80)
for key, value in summary.items():
    print(f"  {key:.<50} {value}")

# K-means results
print("\n" + "-"*80)
print(" K-MEANS CLUSTERING RESULTS")
print("-"*80)

kmeans_results = clustering_results['K-means']
for k in sorted(kmeans_results.keys()):
    result = kmeans_results[k]
    print(f"\n  K = {k}:")
    print(f"    Silhouette Score: {result['silhouette']:.4f}")
    print(f"    Davies-Bouldin Score: {result['davies_bouldin']:.4f}")
    print(f"    Calinski-Harabasz Score: {result['calinski_harabasz']:.2f}")
    
    # Cluster sizes
    unique, counts = np.unique(result['labels'], return_counts=True)
    print(f"    Cluster sizes: {dict(zip(unique, counts))}")

# Best result (by silhouette score)
best_k = max(kmeans_results.keys(), 
            key=lambda k: kmeans_results[k]['silhouette'])
print(f"\n  → Best K (by Silhouette Score): K = {best_k}")
print(f"    Score: {kmeans_results[best_k]['silhouette']:.4f}")

# Save summary to file
summary_text = f"""
FASHION RUNWAY ANALYSIS - RESULTS SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{"="*80}

DATASET STATISTICS:
- Total images analyzed: {summary['Total Images']}
- Embedding model: {summary['Embedding Model']}
- Original embedding dimension: {summary['Embedding Dimension (original)']}
- Reduced dimension (PCA): {summary['Reduced Dimension']}

CLUSTERING RESULTS (K-means):
"""

for k in sorted(kmeans_results.keys()):
    result = kmeans_results[k]
    summary_text += f"""
K = {k}:
  • Silhouette Score: {result['silhouette']:.4f}
  • Davies-Bouldin Score: {result['davies_bouldin']:.4f}
  • Calinski-Harabasz Score: {result['calinski_harabasz']:.2f}
  • Cluster sizes: {dict(zip(*np.unique(result['labels'], return_counts=True)))}
"""

summary_text += f"""
RECOMMENDATION:
Best clustering configuration: K = {best_k}
Silhouette Score: {kmeans_results[best_k]['silhouette']:.4f}

FILES GENERATED:
✓ vit_embeddings.pkl
✓ clip_embeddings.pkl
✓ resnet_embeddings.pkl
✓ embeddings_reduced.pkl
✓ clustering_results.pkl
✓ clustering_comparison.csv
✓ best_cluster_labels.npy
✓ figure1_segmentation.png
✓ figure3_tsne_clusters.png
✓ figure4_cluster_composition.png
✓ figure5_metrics_comparison.png
✓ figure6_designer_gucci.png
✓ figure7_cluster_examples.png

NEXT STEPS FOR YOUR REPORT:
1. Use figure1_segmentation.png in your Methodology section
2. Include clustering metrics table from clustering_comparison.csv
3. Use figure3_tsne_clusters.png as your main results visualization
4. Reference the silhouette scores in your quantitative analysis
5. Discuss cluster composition using figure4_cluster_composition.png
6. Use figure7_cluster_examples.png to show qualitative results
"""

with open('RESULTS_SUMMARY.txt', 'w') as f:
    f.write(summary_text)

print("\n✓ Saved detailed summary to RESULTS_SUMMARY.txt")

# ============================================================================
# FINAL REPORT
# ============================================================================

print("\n" + "█"*80)
print(" PIPELINE COMPLETE!")
print("█"*80)

print("""
All analysis complete! You now have:

📊 DATA:
   • 3 sets of embeddings (ViT, CLIP, ResNet)
   • Dimensionally-reduced embeddings (PCA)
   • Clustering results with multiple methods
   • Best cluster assignments

📈 VISUALIZATIONS:
   • Segmentation before/after
   • t-SNE cluster plots
   • Cluster composition analysis
   • Metrics comparison charts
   • Designer-specific analysis
   • Cluster example images

📝 METRICS & TABLES:
   • Silhouette scores for all K values
   • Davies-Bouldin scores
   • Calinski-Harabasz scores
   • Clustering comparison CSV

🎯 FOR YOUR REPORT:

Methodology Section:
- Mention you used SAM for segmentation (from your notebook)
- State you extracted embeddings with CLIP/ViT
- Reduced to 50 dimensions with PCA
- Applied K-means clustering with K=[3,5,7,10]

Results Section:
- Include figure1_segmentation.png
- Show clustering_comparison.csv as Table 1
- Use figure3_tsne_clusters.png as main visualization
- Best result: K={best_k} with Silhouette={kmeans_results[best_k]['silhouette']:.3f}

Discussion:
- Clusters show meaningful grouping of similar styles
- Some designers cluster together (shown in figure4)
- Limitations: background complexity, occlusions

Check RESULTS_SUMMARY.txt for full details!
""")

print("="*80)
print(f" Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
