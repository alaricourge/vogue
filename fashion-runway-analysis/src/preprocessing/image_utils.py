"""
Utilitaires pour la manipulation d'images
Remplace et améliore les fonctions de utile_vogue.py
"""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Union, List
import matplotlib.pyplot as plt


class ImageHandler:
    """Gestionnaire centralisé pour les opérations sur images"""
    
    @staticmethod
    def load(filepath: Union[str, Path], mode: str = 'RGB') -> Image.Image:
        """
        Charge une image depuis un fichier
        
        Args:
            filepath: Chemin du fichier
            mode: Mode de couleur ('RGB', 'RGBA', 'L' pour grayscale)
            
        Returns:
            Image PIL
        """
        return Image.open(filepath).convert(mode)
    
    @staticmethod
    def load_cv2(filepath: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
        """
        Charge une image avec OpenCV
        
        Args:
            filepath: Chemin du fichier
            flags: Flags OpenCV (IMREAD_COLOR, IMREAD_UNCHANGED, etc.)
            
        Returns:
            Image numpy array (BGR pour color)
        """
        img = cv2.imread(str(filepath), flags)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image: {filepath}")
        return img
    
    @staticmethod
    def save(image: Union[Image.Image, np.ndarray], 
            filepath: Union[str, Path],
            quality: int = 95) -> None:
        """
        Sauvegarde une image
        
        Args:
            image: Image PIL ou numpy array
            filepath: Chemin de destination
            quality: Qualité de compression (1-100) pour JPEG
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(image, np.ndarray):
            # OpenCV array
            cv2.imwrite(str(filepath), image)
        else:
            # PIL Image
            if filepath.suffix.lower() in ['.jpg', '.jpeg']:
                image.save(filepath, quality=quality, optimize=True)
            else:
                image.save(filepath, optimize=True)
    
    @staticmethod
    def resize(image: Union[Image.Image, np.ndarray], 
              size: Tuple[int, int],
              keep_aspect_ratio: bool = False,
              interpolation: int = Image.LANCZOS) -> Union[Image.Image, np.ndarray]:
        """
        Redimensionne une image
        
        Args:
            image: Image PIL ou numpy array
            size: Tuple (width, height)
            keep_aspect_ratio: Si True, préserve le ratio (avec padding)
            interpolation: Méthode d'interpolation
            
        Returns:
            Image redimensionnée du même type que l'entrée
        """
        if isinstance(image, np.ndarray):
            # OpenCV
            if keep_aspect_ratio:
                return ImageHandler._resize_with_aspect_cv2(image, size)
            else:
                interp_cv2 = cv2.INTER_LANCZOS4 if interpolation == Image.LANCZOS else cv2.INTER_LINEAR
                return cv2.resize(image, size, interpolation=interp_cv2)
        else:
            # PIL
            if keep_aspect_ratio:
                image.thumbnail(size, interpolation)
                return image
            else:
                return image.resize(size, interpolation)
    
    @staticmethod
    def _resize_with_aspect_cv2(image: np.ndarray, 
                                target_size: Tuple[int, int]) -> np.ndarray:
        """Redimensionne en préservant le ratio d'aspect (OpenCV)"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculer le ratio
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Redimensionner
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Créer canvas et centrer l'image
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
        
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    @staticmethod
    def to_array(image: Image.Image) -> np.ndarray:
        """Convertit une image PIL en numpy array"""
        return np.array(image)
    
    @staticmethod
    def to_pil(array: np.ndarray, mode: str = 'RGB') -> Image.Image:
        """
        Convertit un numpy array en image PIL
        
        Args:
            array: Numpy array
            mode: Mode de couleur de sortie
            
        Returns:
            Image PIL
        """
        # Si l'array est en BGR (OpenCV), convertir en RGB
        if len(array.shape) == 3 and array.shape[2] == 3 and mode == 'RGB':
            array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(array, mode=mode)
    
    @staticmethod
    def show(image: Union[Image.Image, np.ndarray], 
            title: str = "Image",
            figsize: Tuple[int, int] = (8, 8)) -> None:
        """
        Affiche une image avec matplotlib
        
        Args:
            image: Image à afficher
            title: Titre de la figure
            figsize: Taille de la figure
        """
        plt.figure(figsize=figsize)
        
        if isinstance(image, np.ndarray):
            # Convertir BGR en RGB si nécessaire
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image)
        else:
            plt.imshow(image)
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def get_dimensions(image: Union[Image.Image, np.ndarray]) -> Tuple[int, int]:
        """
        Retourne les dimensions (width, height) de l'image
        
        Args:
            image: Image PIL ou numpy array
            
        Returns:
            Tuple (width, height)
        """
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            return w, h
        else:
            return image.size


def remove_transparent_pixels(image: np.ndarray) -> np.ndarray:
    """
    Retire les pixels transparents d'une image RGBA
    Fonction améliorée de utile_vogue.py
    
    Args:
        image: Image numpy array (peut avoir canal alpha)
        
    Returns:
        Array de pixels non-transparents (sans canal alpha)
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Image RGBA - filtrer les pixels transparents
        alpha_channel = image[:, :, 3]
        non_transparent_mask = alpha_channel > 0
        non_transparent_pixels = image[non_transparent_mask][:, :3]
        return non_transparent_pixels
    else:
        # Image RGB - retourner telle quelle
        return image.reshape(-1, 3) if len(image.shape) == 3 else image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convertit une image BGR (OpenCV) en RGB"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convertit une image RGB en BGR (pour OpenCV)"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
