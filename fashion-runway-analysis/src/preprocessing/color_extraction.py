"""
Extraction de couleurs dominantes depuis des images
Version améliorée de extract_dominant_color de utile_vogue.py
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from PIL import Image

from .image_utils import remove_transparent_pixels, ImageHandler


class ColorExtractor:
    """Extracteur de couleurs dominantes depuis des images"""
    
    def __init__(self, n_colors: int = 5, random_state: int = 42):
        """
        Initialise l'extracteur
        
        Args:
            n_colors: Nombre de couleurs dominantes à extraire
            random_state: Seed pour la reproductibilité
        """
        self.n_colors = n_colors
        self.random_state = random_state
    
    def extract(self, image: Union[np.ndarray, Image.Image, str]) -> List[Tuple[int, int, int]]:
        """
        Extrait les couleurs dominantes d'une image
        
        Args:
            image: Image (numpy array, PIL Image, ou chemin de fichier)
            
        Returns:
            Liste de tuples RGB des couleurs dominantes, triées par fréquence
        """
        # Charger l'image si c'est un chemin
        if isinstance(image, str):
            image = ImageHandler.load(image)
        
        # Convertir PIL en numpy si nécessaire
        if isinstance(image, Image.Image):
            image = ImageHandler.to_array(image)
        
        # Retirer les pixels transparents si présents
        pixels = remove_transparent_pixels(image)
        
        # Reshape pour KMeans (n_pixels, 3)
        if len(pixels.shape) == 3:
            pixels = pixels.reshape(-1, 3)
        
        # Appliquer K-means pour trouver les clusters de couleurs
        kmeans = KMeans(
            n_clusters=self.n_colors, 
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(pixels)
        
        # Compter la fréquence de chaque cluster
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        
        # Créer liste de (couleur, fréquence) et trier par fréquence
        color_freq_pairs = list(zip(kmeans.cluster_centers_, counts))
        color_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extraire les couleurs et convertir en int
        dominant_colors = [
            tuple(int(c) for c in color) 
            for color, freq in color_freq_pairs
        ]
        
        return dominant_colors
    
    def extract_with_frequencies(self, image: Union[np.ndarray, Image.Image, str]) -> List[Tuple[Tuple[int, int, int], int]]:
        """
        Extrait les couleurs dominantes avec leurs fréquences
        
        Args:
            image: Image (numpy array, PIL Image, ou chemin de fichier)
            
        Returns:
            Liste de tuples ((R, G, B), frequency) triés par fréquence
        """
        # Charger et préparer l'image
        if isinstance(image, str):
            image = ImageHandler.load(image)
        if isinstance(image, Image.Image):
            image = ImageHandler.to_array(image)
        
        pixels = remove_transparent_pixels(image)
        if len(pixels.shape) == 3:
            pixels = pixels.reshape(-1, 3)
        
        # K-means
        kmeans = KMeans(
            n_clusters=self.n_colors, 
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(pixels)
        
        # Compter fréquences
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        
        # Trier par fréquence
        color_freq_pairs = list(zip(kmeans.cluster_centers_, counts))
        color_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Convertir en format final
        result = [
            (tuple(int(c) for c in color), int(freq))
            for color, freq in color_freq_pairs
        ]
        
        return result
    
    @staticmethod
    def visualize_palette(colors: List[Tuple[int, int, int]], 
                         figsize: Tuple[int, int] = (10, 2),
                         title: str = "Color Palette") -> None:
        """
        Visualise une palette de couleurs
        
        Args:
            colors: Liste de tuples RGB
            figsize: Taille de la figure
            title: Titre de la visualisation
        """
        n_colors = len(colors)
        fig, axes = plt.subplots(1, n_colors, figsize=figsize)
        
        if n_colors == 1:
            axes = [axes]
        
        for idx, color in enumerate(colors):
            # Créer un carré de couleur
            color_square = np.ones((100, 100, 3), dtype=np.uint8)
            color_square[:, :] = color
            
            axes[idx].imshow(color_square)
            axes[idx].axis('off')
            axes[idx].set_title(f"RGB{color}", fontsize=8)
        
        plt.suptitle(title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def determine_optimal_k(image: Union[np.ndarray, Image.Image], 
                           max_k: int = 10) -> None:
        """
        Utilise la méthode du coude pour déterminer le nombre optimal de couleurs
        Version améliorée de numbers_dominant_color de utile_vogue.py
        
        Args:
            image: Image à analyser
            max_k: Nombre maximum de clusters à tester
        """
        if isinstance(image, Image.Image):
            image = ImageHandler.to_array(image)
        
        pixels = remove_transparent_pixels(image)
        if len(pixels.shape) == 3:
            pixels = pixels.reshape(-1, 3)
        
        inertias = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            inertias.append(kmeans.inertia_)
        
        # Visualiser
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Colors (K)', fontsize=12)
        plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def extract_dominant_color(image: Union[np.ndarray, Image.Image, str], 
                          k: int = 5) -> List[Tuple[int, int, int]]:
    """
    Fonction utilitaire pour extraction rapide de couleurs dominantes
    Compatible avec l'ancienne API de utile_vogue.py
    
    Args:
        image: Image à analyser
        k: Nombre de couleurs à extraire
        
    Returns:
        Liste de tuples RGB des couleurs dominantes
    """
    extractor = ColorExtractor(n_colors=k)
    return extractor.extract(image)
