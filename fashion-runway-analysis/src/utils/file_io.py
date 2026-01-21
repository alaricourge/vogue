"""
Utilitaires pour la gestion des fichiers et I/O
"""
import os
import pickle
import json
from pathlib import Path
from typing import List, Union, Any
import pandas as pd


def get_image_files(folder: Union[str, Path], 
                   extensions: tuple = ('.png', '.jpg', '.jpeg')) -> List[Path]:
    """
    Récupère tous les fichiers images d'un dossier
    
    Args:
        folder: Chemin du dossier
        extensions: Tuple des extensions acceptées
        
    Returns:
        Liste triée des chemins d'images
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dossier non trouvé: {folder}")
    
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        # Gérer aussi les majuscules
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    return sorted(set(image_files))


def save_pickle(data: Any, filepath: Union[str, Path]) -> None:
    """
    Sauvegarde des données au format pickle
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin du fichier de sortie
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Charge des données depuis un fichier pickle
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Données chargées
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(data: dict, filepath: Union[str, Path], indent: int = 2) -> None:
    """
    Sauvegarde des données au format JSON
    
    Args:
        data: Dictionnaire à sauvegarder
        filepath: Chemin du fichier de sortie
        indent: Indentation pour le formatage
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> dict:
    """
    Charge des données depuis un fichier JSON
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Dictionnaire chargé
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
    """
    Sauvegarde un DataFrame au format CSV
    
    Args:
        df: DataFrame à sauvegarder
        filepath: Chemin du fichier de sortie
        **kwargs: Arguments additionnels pour pd.to_csv
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(filepath, **kwargs)


def load_csv(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Charge un fichier CSV dans un DataFrame
    
    Args:
        filepath: Chemin du fichier
        **kwargs: Arguments additionnels pour pd.read_csv
        
    Returns:
        DataFrame chargé
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
    
    return pd.read_csv(filepath, **kwargs)


def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    S'assure qu'un répertoire existe, le crée si nécessaire
    
    Args:
        directory: Chemin du répertoire
        
    Returns:
        Path object du répertoire
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_file_size_mb(filepath: Union[str, Path]) -> float:
    """
    Retourne la taille d'un fichier en Mo
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Taille en mégaoctets
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return 0.0
    
    return filepath.stat().st_size / (1024 * 1024)


def count_files(directory: Union[str, Path], 
               extensions: tuple = None) -> int:
    """
    Compte le nombre de fichiers dans un répertoire
    
    Args:
        directory: Chemin du répertoire
        extensions: Tuple d'extensions à filtrer (None = tous)
        
    Returns:
        Nombre de fichiers
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    
    if extensions is None:
        return len(list(directory.iterdir()))
    
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f"*{ext}")))
        count += len(list(directory.glob(f"*{ext.upper()}")))
    
    return count
