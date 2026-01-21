"""
Extraction d'embeddings visuels pour les images de défilés
Version modulaire et améliorée de embeddings_extraction.py
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
import warnings
warnings.filterwarnings('ignore')

from ..utils.file_io import get_image_files, save_pickle
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingExtractor:
    """
    Extracteur d'embeddings visuels unifié pour plusieurs modèles
    """
    
    SUPPORTED_MODELS = {
        'vit': {
            'name': 'Vision Transformer',
            'model_id': 'google/vit-base-patch16-224',
            'embedding_dim': 768
        },
        'clip': {
            'name': 'CLIP',
            'model_id': 'openai/clip-vit-base-patch32',
            'embedding_dim': 512
        },
        'resnet': {
            'name': 'ResNet50',
            'model_id': 'resnet50',
            'embedding_dim': 2048
        }
    }
    
    def __init__(self, model_type: str = 'clip', device: str = None):
        """
        Initialise l'extracteur
        
        Args:
            model_type: Type de modèle ('vit', 'clip', 'resnet')
            device: Device à utiliser ('cuda', 'cpu', None pour auto-détection)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' non supporté. "
                           f"Choisir parmi: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_info = self.SUPPORTED_MODELS[model_type]
        
        # Déterminer le device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initialisation du modèle {self.model_info['name']} sur {self.device}")
        
        # Charger le modèle
        self.model, self.processor = self._load_model()
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model(self):
        """Charge le modèle et son processeur selon le type"""
        if self.model_type == 'vit':
            from transformers import ViTImageProcessor, ViTModel
            processor = ViTImageProcessor.from_pretrained(self.model_info['model_id'])
            model = ViTModel.from_pretrained(self.model_info['model_id'])
            return model, processor
        
        elif self.model_type == 'clip':
            from transformers import CLIPProcessor, CLIPModel
            processor = CLIPProcessor.from_pretrained(self.model_info['model_id'])
            model = CLIPModel.from_pretrained(self.model_info['model_id'])
            return model, processor
        
        elif self.model_type == 'resnet':
            from torchvision import models, transforms
            model = models.resnet50(weights='IMAGENET1K_V1')
            # Retirer la couche de classification finale
            model = torch.nn.Sequential(*list(model.children())[:-1])
            
            # Définir les transformations
            processor = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            return model, processor
    
    def _extract_single(self, image_path: Path) -> np.ndarray:
        """
        Extrait l'embedding d'une seule image
        
        Args:
            image_path: Chemin de l'image
            
        Returns:
            Embedding numpy array
        """
        # Charger l'image
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            if self.model_type == 'vit':
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                # Utiliser le token CLS
                embedding = outputs.last_hidden_state[:, 0].cpu().numpy()
                return embedding.flatten()
            
            elif self.model_type == 'clip':
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                vision_outputs = self.model.get_image_features(**inputs)
                return vision_outputs.cpu().numpy().flatten()
            
            elif self.model_type == 'resnet':
                img_tensor = self.processor(image).unsqueeze(0).to(self.device)
                features = self.model(img_tensor)
                return features.squeeze().cpu().numpy()
    
    def extract(self, 
               image_folder: Union[str, Path],
               extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
               batch_size: int = 1) -> Dict[str, any]:
        """
        Extrait les embeddings de toutes les images d'un dossier
        
        Args:
            image_folder: Dossier contenant les images
            extensions: Extensions de fichiers acceptées
            batch_size: Taille du batch (pour optimisation future)
            
        Returns:
            Dictionnaire avec embeddings, chemins, et métadonnées
        """
        image_folder = Path(image_folder)
        
        # Récupérer les fichiers images
        image_files = get_image_files(image_folder, extensions)
        
        if not image_files:
            raise ValueError(f"Aucune image trouvée dans {image_folder}")
        
        logger.info(f"Extraction de {len(image_files)} embeddings {self.model_info['name']}...")
        
        embeddings = []
        image_paths = []
        failed = []
        
        # Extraire les embeddings
        for img_path in tqdm(image_files, desc=f"Extracting {self.model_type}"):
            try:
                embedding = self._extract_single(img_path)
                embeddings.append(embedding)
                image_paths.append(str(img_path))
            except Exception as e:
                logger.warning(f"Erreur avec {img_path.name}: {e}")
                failed.append(str(img_path))
        
        embeddings = np.array(embeddings)
        
        logger.info(f"✓ {len(embeddings)} embeddings extraits avec succès")
        logger.info(f"  Dimension: {embeddings.shape[1]}")
        
        if failed:
            logger.warning(f"  {len(failed)} images ont échoué")
        
        return {
            'embeddings': embeddings,
            'image_paths': image_paths,
            'model_type': self.model_type,
            'model_name': self.model_info['name'],
            'model_id': self.model_info['model_id'],
            'embedding_dim': self.model_info['embedding_dim'],
            'failed_images': failed
        }
    
    def extract_and_save(self, 
                        image_folder: Union[str, Path],
                        output_file: Union[str, Path],
                        **kwargs) -> Dict[str, any]:
        """
        Extrait et sauvegarde les embeddings
        
        Args:
            image_folder: Dossier des images
            output_file: Fichier de sortie (.pkl)
            **kwargs: Arguments pour extract()
            
        Returns:
            Résultats de l'extraction
        """
        results = self.extract(image_folder, **kwargs)
        save_pickle(results, output_file)
        logger.info(f"✓ Embeddings sauvegardés dans {output_file}")
        return results
    
    @classmethod
    def extract_all_models(cls, 
                          image_folder: Union[str, Path],
                          output_dir: Union[str, Path],
                          models: List[str] = None,
                          device: str = None) -> Dict[str, Dict]:
        """
        Extrait les embeddings avec tous les modèles
        
        Args:
            image_folder: Dossier des images
            output_dir: Dossier de sortie
            models: Liste des modèles à utiliser (None = tous)
            device: Device à utiliser
            
        Returns:
            Dictionnaire des résultats pour chaque modèle
        """
        if models is None:
            models = list(cls.SUPPORTED_MODELS.keys())
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for model_type in models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Modèle: {cls.SUPPORTED_MODELS[model_type]['name']}")
            logger.info(f"{'='*60}")
            
            extractor = cls(model_type=model_type, device=device)
            output_file = output_dir / f"{model_type}_embeddings.pkl"
            
            results = extractor.extract_and_save(
                image_folder=image_folder,
                output_file=output_file
            )
            
            all_results[model_type] = results
        
        return all_results


def load_embeddings(filepath: Union[str, Path]) -> Dict[str, any]:
    """
    Charge des embeddings depuis un fichier pickle
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Dictionnaire des embeddings et métadonnées
    """
    from ..utils.file_io import load_pickle
    return load_pickle(filepath)
