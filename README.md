# Fashion Runway Analysis - Quick Start Guide

## Installation Requirements

```bash
# Install required packages
pip install torch torchvision --break-system-packages
pip install transformers --break-system-packages
pip install scikit-learn --break-system-packages
pip install pandas numpy matplotlib seaborn --break-system-packages
pip install Pillow opencv-python --break-system-packages
pip install tqdm --break-system-packages
```

## Required Python Packages

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0
opencv-python>=4.8.0
tqdm>=4.65.0
```

## File Structure

```
your_project/
├── defile_vogue/
│   ├── non_traitee/          # Original images
│   ├── traitee/              # Preprocessed images
│   └── tres_traitee/         # Segmented images (from your SAM code)
├── information_defile.csv    # Metadata (designer, season, year)
├── utile_vogue.py            # Your utility functions
├── embeddings_extraction.py  # Extract visual embeddings
├── clustering_analysis.py    # Perform clustering
├── visualization.py          # Generate figures
└── run_pipeline.py          # Master script
```

## Quick Start (3 Options)

### OPTION 1: Run Everything at Once (Recommended)
```bash
python run_pipeline.py
```
This will:
1. Extract embeddings (ViT, CLIP, ResNet) - ~20-30 min
2. Perform clustering analysis - ~5 min
3. Generate all visualizations - ~10 min
4. Create results summary

### OPTION 2: Run Step-by-Step
```bash
# Step 1: Extract embeddings
python embeddings_extraction.py

# Step 2: Cluster the embeddings
python clustering_analysis.py

# Step 3: Create visualizations
python visualization.py
```

### OPTION 3: Use in Jupyter Notebook
```python
# In your notebook:
import embeddings_extraction
import clustering_analysis
import visualization

# Extract embeddings
embeddings, paths = embeddings_extraction.extract_clip_embeddings(
    'defile_vogue/tres_traitee'
)

# Cluster
from clustering_analysis import kmeans_clustering, reduce_dimensions
emb_reduced, pca = reduce_dimensions(embeddings, n_components=50)
results = kmeans_clustering(emb_reduced, k_values=[3, 5, 7, 10])

# Visualize
from visualization import visualize_tsne_clusters
visualize_tsne_clusters(emb_reduced, results[5]['labels'])
```

## Expected Output Files

### Data Files:
- `vit_embeddings.pkl` - Vision Transformer features
- `clip_embeddings.pkl` - CLIP features (recommended)
- `resnet_embeddings.pkl` - ResNet features (baseline)
- `embeddings_reduced.pkl` - PCA-reduced embeddings
- `clustering_results.pkl` - All clustering results
- `best_cluster_labels.npy` - Best cluster assignments

### Visualization Files:
- `figure1_segmentation.png` - Before/after segmentation
- `figure2_color_palettes.png` - Color extraction examples
- `figure3_tsne_clusters.png` - Main cluster visualization
- `figure4_cluster_composition.png` - Cluster analysis by designer/season
- `figure5_metrics_comparison.png` - Clustering metrics
- `figure6_designer_gucci.png` - Designer-specific clustering
- `figure7_cluster_examples.png` - Sample images from each cluster

### Result Files:
- `clustering_comparison.csv` - Metrics table for your report
- `RESULTS_SUMMARY.txt` - Complete results summary

## Usage Notes

### 1. Adjust File Paths
Update these in the scripts if your paths differ:
```python
# In embeddings_extraction.py
IMAGE_FOLDER = "defile_vogue/tres_traitee"

# In visualization.py (main section)
original_folder='defile_vogue/traitee'
segmented_folder='defile_vogue/tres_traitee'
```

### 2. Choose Embedding Model
The code extracts 3 types of embeddings:
- **CLIP** (recommended) - Best for fashion, understands semantics
- **ViT** - Good alternative, similar performance
- **ResNet** - Baseline, faster but less powerful

For your report, use CLIP or ViT.

### 3. Select Optimal K
The code tests K=[3, 5, 7, 10]. Based on silhouette scores:
- K=5 or K=7 typically work best for fashion
- Higher silhouette score = better clustering
- Check `clustering_comparison.csv` for exact values

### 4. GPU vs CPU
- Code automatically uses GPU if available
- GPU recommended (20 min vs 2+ hours for embeddings)
- If no GPU: reduce dataset size for testing

## For Your Report

### Methodology Section:
```
We extracted visual embeddings using CLIP (Radford et al., 2021), 
a vision-language model pretrained on 400M image-text pairs. Each 
segmented runway image was encoded into a 512-dimensional vector 
representing its visual and semantic content. 

To reduce computational complexity and mitigate the curse of 
dimensionality, we applied PCA to reduce embeddings to 50 dimensions 
while preserving 95% of variance.

We then applied K-means clustering with K ∈ {3, 5, 7, 10}, evaluating 
each configuration using silhouette score, Davies-Bouldin index, and 
Calinski-Harabasz score.
```

### Results Section:
```
Table 1 shows clustering metrics for different K values. We achieved 
the best silhouette score of [X.XXX] with K=[Y], indicating well-
separated, cohesive clusters.

Figure 3 visualizes the clusters using t-SNE dimensionality reduction. 
The separation between clusters suggests that our embeddings successfully 
capture meaningful stylistic differences.

Figure 4 analyzes cluster composition by designer, revealing that 
[finding - e.g., "Cluster 2 predominantly contains Gucci and Off-White 
pieces, suggesting shared streetwear influences"].
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision --break-system-packages
```

### "CUDA out of memory"
Reduce batch size or use CPU:
```python
device = "cpu"  # In embeddings_extraction.py
```

### "No such file or directory"
Check your paths:
```python
import os
print(os.listdir('defile_vogue'))  # Should show your folders
```

### Images not loading
Ensure image extensions match:
```python
# The code looks for .png, .jpg, .jpeg
# Make sure your files have these extensions
```
