"""
Fashion Runway Analysis - Pipeline Principal
Point d'entrée unique pour tout le projet
"""
import argparse
from pathlib import Path
import sys

from config import config
from src.utils.logger import setup_logger, log_section, log_step
from src.models.embeddings import EmbeddingExtractor
from src.utils.file_io import count_files

logger = setup_logger("FashionRunway")


def step_extract_embeddings(args):
    """Étape 1: Extraction des embeddings"""
    log_section(logger, "ÉTAPE 1: EXTRACTION DES EMBEDDINGS")
    
    # Vérifier que le dossier d'images existe
    if not config.SEGMENTED_224.exists():
        logger.error(f"Dossier d'images introuvable: {config.SEGMENTED_224}")
        logger.info("Veuillez d'abord segmenter les images avec:")
        logger.info("  python main.py --segment-images")
        return False
    
    # Compter les images
    n_images = count_files(config.SEGMENTED_224, config.IMAGE_EXTENSIONS)
    logger.info(f"Images trouvées: {n_images}")
    
    if n_images == 0:
        logger.error("Aucune image trouvée!")
        return False
    
    # Extraire les embeddings
    models_to_use = args.models if args.models else ['clip']
    logger.info(f"Modèles à utiliser: {', '.join(models_to_use)}")
    
    try:
        EmbeddingExtractor.extract_all_models(
            image_folder=config.SEGMENTED_224,
            output_dir=config.EMBEDDINGS_DIR,
            models=models_to_use,
            device=config.DEVICE
        )
        logger.info("✅ Extraction des embeddings terminée!")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction: {e}")
        return False


def step_clustering(args):
    """Étape 2: Analyse de clustering"""
    log_section(logger, "ÉTAPE 2: ANALYSE DE CLUSTERING")
    
    # Vérifier que les embeddings existent
    embedding_file = config.get_embedding_path('clip')
    if not embedding_file.exists():
        logger.error(f"Fichier d'embeddings introuvable: {embedding_file}")
        logger.info("Veuillez d'abord extraire les embeddings avec:")
        logger.info("  python main.py --extract-embeddings")
        return False
    
    try:
        # Import ici pour éviter les dépendances si non utilisé
        from src.models.clustering import ClusterAnalyzer
        
        analyzer = ClusterAnalyzer()
        results = analyzer.run_all_methods(
            embeddings_file=embedding_file,
            k_values=config.K_VALUES,
            output_dir=config.CLUSTERING_DIR
        )
        
        logger.info("✅ Analyse de clustering terminée!")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du clustering: {e}")
        return False


def step_visualization(args):
    """Étape 3: Génération des visualisations"""
    log_section(logger, "ÉTAPE 3: GÉNÉRATION DES VISUALISATIONS")
    
    # Vérifier que les résultats existent
    clustering_file = config.CLUSTERING_DIR / "kmeans_results.pkl"
    if not clustering_file.exists():
        logger.error("Résultats de clustering introuvables!")
        logger.info("Veuillez d'abord effectuer le clustering avec:")
        logger.info("  python main.py --cluster")
        return False
    
    try:
        # Import ici pour éviter les dépendances si non utilisé
        from src.visualization.plots import ResultsVisualizer
        
        visualizer = ResultsVisualizer()
        visualizer.generate_all_figures(
            clustering_results_file=clustering_file,
            output_dir=config.FIGURES_DIR
        )
        
        logger.info("✅ Visualisations générées!")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la visualisation: {e}")
        return False


def step_full_pipeline(args):
    """Exécute le pipeline complet"""
    log_section(logger, "PIPELINE COMPLET - FASHION RUNWAY ANALYSIS", char="█")
    
    steps = [
        ("Extraction des embeddings", step_extract_embeddings),
        ("Analyse de clustering", step_clustering),
        ("Génération des visualisations", step_visualization),
    ]
    
    for i, (name, step_func) in enumerate(steps, 1):
        log_step(logger, i, len(steps), name)
        
        success = step_func(args)
        if not success:
            logger.error(f"❌ Le pipeline s'est arrêté à l'étape {i}")
            return False
    
    log_section(logger, "✅ PIPELINE TERMINÉ AVEC SUCCÈS!", char="█")
    return True


def step_segment_images(args):
    """Étape 0 (optionnelle): Segmentation des images"""
    log_section(logger, "ÉTAPE 0: SEGMENTATION DES IMAGES")
    
    logger.info("⚠️  La segmentation n'est pas implémentée dans ce script")
    logger.info("Utilisez votre notebook de segmentation SAM existant")
    logger.info("et placez les images segmentées dans:")
    logger.info(f"  {config.SEGMENTED_512}")
    logger.info(f"  {config.SEGMENTED_224}")
    return False


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Fashion Runway Analysis - Pipeline d'analyse de défilés",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Pipeline complet
  python main.py --full
  
  # Extraction d'embeddings uniquement
  python main.py --extract-embeddings
  
  # Extraction avec plusieurs modèles
  python main.py --extract-embeddings --models clip vit resnet
  
  # Clustering uniquement
  python main.py --cluster
  
  # Visualisation uniquement
  python main.py --visualize
  
  # Étapes individuelles
  python main.py --extract-embeddings --cluster
        """
    )
    
    # Arguments principaux
    parser.add_argument('--full', action='store_true',
                       help='Exécuter le pipeline complet')
    parser.add_argument('--extract-embeddings', action='store_true',
                       help='Extraire les embeddings')
    parser.add_argument('--cluster', action='store_true',
                       help='Effectuer le clustering')
    parser.add_argument('--visualize', action='store_true',
                       help='Générer les visualisations')
    parser.add_argument('--segment-images', action='store_true',
                       help='Segmenter les images (guide uniquement)')
    
    # Arguments optionnels
    parser.add_argument('--models', nargs='+', 
                       choices=['clip', 'vit', 'resnet'],
                       help='Modèles pour extraction d\'embeddings')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device à utiliser (auto-détection par défaut)')
    
    args = parser.parse_args()
    
    # Si aucun argument, afficher l'aide
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Configurer le device si spécifié
    if args.device:
        config.DEVICE = args.device
    
    # Créer les répertoires nécessaires
    config.ensure_directories()
    
    # Exécuter les étapes demandées
    try:
        if args.full:
            step_full_pipeline(args)
        else:
            if args.segment_images:
                step_segment_images(args)
            
            if args.extract_embeddings:
                step_extract_embeddings(args)
            
            if args.cluster:
                step_clustering(args)
            
            if args.visualize:
                step_visualization(args)
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
