"""
Embedding Extraction for Fashion Runway Images
Extracts deep visual features using Vision Transformers (ViT) and CLIP
"""

import torch
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import pickle

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================================
# METHOD 1: Vision Transformer (ViT) Embeddings
# ============================================================================

def extract_vit_embeddings(image_folder, output_file='vit_embeddings.pkl'):
    """
    Extract ViT embeddings for all images in a folder
    
    Args:
        image_folder: Path to folder with segmented images
        output_file: Where to save the embeddings
    
    Returns:
        embeddings: numpy array of shape (n_images, embedding_dim)
        image_paths: list of image paths corresponding to embeddings
    """
    from transformers import ViTImageProcessor, ViTModel
    
    print("Loading ViT model...")
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    model.to(device)
    model.eval()
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    embeddings = []
    image_paths = []
    
    print(f"Extracting embeddings for {len(image_files)} images...")
    
    with torch.no_grad():
        for img_file in tqdm(image_files):
            try:
                img_path = os.path.join(image_folder, img_file)
                
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract features
                outputs = model(**inputs)
                
                # Use CLS token as image embedding
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                
                embeddings.append(embedding.flatten())
                image_paths.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    embeddings = np.array(embeddings)
    
    # Save embeddings
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'image_paths': image_paths,
            'model': 'ViT-base-patch16-224'
        }, f)
    
    print(f"✓ Saved {len(embeddings)} ViT embeddings to {output_file}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, image_paths


# ============================================================================
# METHOD 2: CLIP Embeddings (Vision-Language)
# ============================================================================

def extract_clip_embeddings(image_folder, output_file='clip_embeddings.pkl'):
    """
    Extract CLIP visual embeddings for all images
    CLIP provides better semantic understanding for fashion
    
    Args:
        image_folder: Path to folder with segmented images
        output_file: Where to save the embeddings
    
    Returns:
        embeddings: numpy array of shape (n_images, 512)
        image_paths: list of image paths
    """
    from transformers import CLIPProcessor, CLIPModel
    
    print("Loading CLIP model...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    model.eval()
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    embeddings = []
    image_paths = []
    
    print(f"Extracting CLIP embeddings for {len(image_files)} images...")
    
    with torch.no_grad():
        for img_file in tqdm(image_files):
            try:
                img_path = os.path.join(image_folder, img_file)
                
                # Load image
                image = Image.open(img_path).convert('RGB')
                
                # Process image
                inputs = processor(images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Extract vision features
                vision_outputs = model.get_image_features(**inputs)
                embedding = vision_outputs.cpu().numpy()
                
                embeddings.append(embedding.flatten())
                image_paths.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    embeddings = np.array(embeddings)
    
    # Save embeddings
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'image_paths': image_paths,
            'model': 'CLIP-ViT-base-patch32'
        }, f)
    
    print(f"✓ Saved {len(embeddings)} CLIP embeddings to {output_file}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, image_paths


# ============================================================================
# METHOD 3: ResNet Embeddings (Baseline)
# ============================================================================

def extract_resnet_embeddings(image_folder, output_file='resnet_embeddings.pkl'):
    """
    Extract ResNet50 embeddings as a CNN baseline
    
    Args:
        image_folder: Path to folder with segmented images
        output_file: Where to save the embeddings
    
    Returns:
        embeddings: numpy array of shape (n_images, 2048)
        image_paths: list of image paths
    """
    from torchvision import models, transforms
    
    print("Loading ResNet50 model...")
    model = models.resnet50(pretrained=True)
    # Remove the final classification layer
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    embeddings = []
    image_paths = []
    
    print(f"Extracting ResNet embeddings for {len(image_files)} images...")
    
    with torch.no_grad():
        for img_file in tqdm(image_files):
            try:
                img_path = os.path.join(image_folder, img_file)
                
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Extract features
                features = model(img_tensor)
                embedding = features.squeeze().cpu().numpy()
                
                embeddings.append(embedding)
                image_paths.append(img_path)
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
    
    embeddings = np.array(embeddings)
    
    # Save embeddings
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings,
            'image_paths': image_paths,
            'model': 'ResNet50'
        }, f)
    
    print(f"✓ Saved {len(embeddings)} ResNet embeddings to {output_file}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, image_paths


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_embeddings(embedding_file):
    """Load saved embeddings from pickle file"""
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['image_paths'], data['model']


def combine_color_and_embeddings(embeddings, color_features):
    """
    Combine deep embeddings with color features
    
    Args:
        embeddings: (n_samples, embed_dim)
        color_features: (n_samples, color_dim) - from your K-means color extraction
    
    Returns:
        combined: (n_samples, embed_dim + color_dim)
    """
    # Normalize both features to have similar scales
    from sklearn.preprocessing import StandardScaler
    
    scaler_embed = StandardScaler()
    scaler_color = StandardScaler()
    
    embeddings_norm = scaler_embed.fit_transform(embeddings)
    color_norm = scaler_color.fit_transform(color_features)
    
    # Concatenate
    combined = np.concatenate([embeddings_norm, color_norm], axis=1)
    
    return combined


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    IMAGE_FOLDER = "defile_vogue/tres_traitee"  # Your segmented images
    
    print("="*60)
    print("FASHION RUNWAY EMBEDDING EXTRACTION")
    print("="*60)
    
    # Extract embeddings using all three methods
    # You can choose which one(s) to use for your report
    
    print("\n[1/3] Extracting ViT embeddings...")
    vit_embeddings, vit_paths = extract_vit_embeddings(
        IMAGE_FOLDER, 
        'vit_embeddings.pkl'
    )
    
    print("\n[2/3] Extracting CLIP embeddings...")
    clip_embeddings, clip_paths = extract_clip_embeddings(
        IMAGE_FOLDER, 
        'clip_embeddings.pkl'
    )
    
    print("\n[3/3] Extracting ResNet embeddings...")
    resnet_embeddings, resnet_paths = extract_resnet_embeddings(
        IMAGE_FOLDER, 
        'resnet_embeddings.pkl'
    )
    
    print("\n" + "="*60)
    print("✓ ALL EMBEDDINGS EXTRACTED SUCCESSFULLY!")
    print("="*60)
    print(f"\nViT:    {vit_embeddings.shape}")
    print(f"CLIP:   {clip_embeddings.shape}")
    print(f"ResNet: {resnet_embeddings.shape}")
    print("\nFiles saved:")
    print("  - vit_embeddings.pkl")
    print("  - clip_embeddings.pkl")
    print("  - resnet_embeddings.pkl")
