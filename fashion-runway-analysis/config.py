"""
Configuration centralisée pour le projet Fashion Runway Analysis
VERSION WINDOWS avec chemins absolus
"""
from pathlib import Path

class Config:
    """Configuration globale du projet"""
    
    # Chemins de base
    PROJECT_ROOT = Path(__file__).parent
    
    # ⚠️ CHEMINS ABSOLUS VERS VOS IMAGES
    # Modifiez ces chemins selon votre configuration
    DATA_DIR = Path(r"C:\home\claude\data")
    RAW_IMAGES = DATA_DIR / "raw"
    
    # Images téléchargées par le scraper
    SEGMENTED_512 = RAW_IMAGES / "downloaded_512"
    SEGMENTED_224 = RAW_IMAGES / "downloaded_224"
    
    # Dossiers de sortie (dans le projet)
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # Autres dossiers (dans le projet)
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    THUMBNAILS = PROCESSED_DIR / "thumbnails"
    
    EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
    RESULTS_DIR = PROJECT_ROOT / "data" / "results"
    CLUSTERING_DIR = RESULTS_DIR / "clustering"
    METADATA_DIR = RESULTS_DIR / "metadata"
    
    # Sorties
    FIGURES_DIR = OUTPUTS_DIR / "figures"
    REPORTS_DIR = OUTPUTS_DIR / "reports"
    
    # Paramètres d'images
    IMAGE_SIZE_HIGH = (512, 512)      # Haute qualité pour visualisation
    IMAGE_SIZE_EMBEDDING = (224, 224)  # Taille pour les modèles
    IMAGE_SIZE_THUMBNAIL = (128, 128)  # Pour aperçus rapides
    
    # Extensions d'images supportées
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    # Modèles d'embeddings
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    VIT_MODEL = "google/vit-base-patch16-224"
    RESNET_MODEL = "resnet50"
    
    # Paramètres de clustering
    K_VALUES = [3, 5, 7, 10]
    PCA_COMPONENTS = 50
    DBSCAN_EPS_VALUES = [1.0, 2.0, 3.0]
    DBSCAN_MIN_SAMPLES = 3
    
    # Paramètres de traitement
    BATCH_SIZE = 32
    RANDOM_SEED = 42
    
    # Device (sera auto-détecté)
    DEVICE = "cuda"  # Changera automatiquement à "cpu" si CUDA non disponible
    
    # Scraping (Vogue)
    VOGUE_BASE_URL = "https://www.vogue.com"
    REQUEST_TIMEOUT = 30
    REQUEST_DELAY = 1  # Délai entre requêtes (secondes)
    
    # Visualisation
    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    COLOR_PALETTE = 'husl'
    
    @classmethod
    def ensure_directories(cls):
        """Crée tous les répertoires nécessaires s'ils n'existent pas"""
        directories = [
            # Ne pas créer les dossiers d'images (ils existent déjà)
            # cls.RAW_IMAGES,
            # cls.SEGMENTED_512,
            # cls.SEGMENTED_224,
            cls.THUMBNAILS,
            cls.EMBEDDINGS_DIR,
            cls.CLUSTERING_DIR,
            cls.METADATA_DIR,
            cls.FIGURES_DIR,
            cls.REPORTS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_embedding_path(cls, model_name):
        """Retourne le chemin pour sauvegarder un fichier d'embeddings"""
        return cls.EMBEDDINGS_DIR / f"{model_name}_embeddings.pkl"
    
    @classmethod
    def get_clustering_path(cls, method_name):
        """Retourne le chemin pour sauvegarder les résultats de clustering"""
        return cls.CLUSTERING_DIR / f"{method_name}_results.pkl"
    
    @classmethod
    def get_figure_path(cls, figure_name):
        """Retourne le chemin pour sauvegarder une figure"""
        if not figure_name.endswith(f'.{cls.FIGURE_FORMAT}'):
            figure_name = f"{figure_name}.{cls.FIGURE_FORMAT}"
        return cls.FIGURES_DIR / figure_name


# Instance globale de configuration
config = Config()
