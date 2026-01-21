"""
Configuration du système de logging
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "FashionRunway", 
                level: int = logging.INFO,
                log_file: str = None) -> logging.Logger:
    """
    Configure et retourne un logger
    
    Args:
        name: Nom du logger
        level: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin optionnel pour sauvegarder les logs
        
    Returns:
        Logger configuré
    """
    # Créer le logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter les handlers multiples si le logger existe déjà
    if logger.handlers:
        return logger
    
    # Format des messages
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler pour la console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Handler pour le fichier (optionnel)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_section(logger: logging.Logger, title: str, char: str = "=", width: int = 80):
    """
    Log une section avec un titre encadré
    
    Args:
        logger: Logger à utiliser
        title: Titre de la section
        char: Caractère de décoration
        width: Largeur de la ligne
    """
    logger.info(char * width)
    logger.info(f" {title}")
    logger.info(char * width)


def log_step(logger: logging.Logger, step: int, total: int, description: str):
    """
    Log une étape du pipeline
    
    Args:
        logger: Logger à utiliser
        step: Numéro de l'étape actuelle
        total: Nombre total d'étapes
        description: Description de l'étape
    """
    logger.info(f"[{step}/{total}] {description}")


def log_results(logger: logging.Logger, results: dict, title: str = "Results"):
    """
    Log un dictionnaire de résultats de manière formatée
    
    Args:
        logger: Logger à utiliser
        results: Dictionnaire de résultats
        title: Titre de la section
    """
    logger.info(f"\n{title}:")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def log_file_info(logger: logging.Logger, filepath: Path, description: str = "File"):
    """
    Log des informations sur un fichier
    
    Args:
        logger: Logger à utiliser
        filepath: Chemin du fichier
        description: Description du fichier
    """
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ {description}: {filepath.name} ({size_mb:.2f} MB)")
    else:
        logger.warning(f"✗ {description}: {filepath.name} (not found)")


# Logger global par défaut
default_logger = setup_logger()
