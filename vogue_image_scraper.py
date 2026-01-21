#!/usr/bin/env python3
"""
Vogue Image Scraper
Télécharge les images depuis le CSV et les stocke en 512x512 et 224x224
"""

import csv
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import time
import requests
from PIL import Image
from io import BytesIO


class VogueImageScraper:
    """Scraper pour télécharger et redimensionner les images Vogue"""
    
    def __init__(self, csv_path: str, output_base: str = "data", max_images: int = 5000):
        """
        Initialise le scraper
        
        Args:
            csv_path: Chemin vers le CSV source
            output_base: Dossier de base pour les données
            max_images: Nombre maximum d'images à télécharger
        """
        self.csv_path = csv_path
        self.max_images = max_images
        
        # Créer les dossiers de sortie
        self.output_512 = Path(output_base) / "raw" / "downloaded_512"
        self.output_224 = Path(output_base) / "raw" / "downloaded_224"
        
        self.output_512.mkdir(parents=True, exist_ok=True)
        self.output_224.mkdir(parents=True, exist_ok=True)
        
        # Statistiques
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # Configuration requêtes HTTP
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def generate_filename(self, designer: str, saison: str, annee: str, num: str) -> str:
        """
        Génère un nom de fichier unique et descriptif
        
        Args:
            designer: Nom du designer
            saison: Saison du défilé
            annee: Année
            num: Numéro de l'image
            
        Returns:
            Nom de fichier formaté
        """
        # Nettoyer les noms
        designer_clean = designer.replace(' ', '-').lower()
        saison_clean = saison.replace(' ', '-').replace('--', '-').lower()
        
        return f"{designer_clean}_{saison_clean}_{annee}_{num}.jpg"
    
    def download_and_resize(self, url: str, filename: str) -> bool:
        """
        Télécharge une image et la sauvegarde en 2 tailles
        
        Args:
            url: URL de l'image
            filename: Nom de fichier de sortie
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Vérifier si l'image existe déjà
            path_512 = self.output_512 / filename
            path_224 = self.output_224 / filename
            
            if path_512.exists() and path_224.exists():
                self.stats['skipped'] += 1
                return True
            
            # Télécharger l'image
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Ouvrir avec PIL
            img = Image.open(BytesIO(response.content))
            
            # Convertir en RGB si nécessaire (pour éviter les problèmes avec RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionner et sauvegarder en 512x512
            img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
            img_512.save(path_512, 'JPEG', quality=95)
            
            # Redimensionner et sauvegarder en 224x224
            img_224 = img.resize((224, 224), Image.Resampling.LANCZOS)
            img_224.save(path_224, 'JPEG', quality=95)
            
            self.stats['successful'] += 1
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Erreur réseau pour {url}: {e}")
            self.stats['failed'] += 1
            return False
            
        except Exception as e:
            print(f"\n❌ Erreur traitement pour {url}: {e}")
            self.stats['failed'] += 1
            return False
    
    def scrape(self):
        """Lance le scraping depuis le CSV"""
        
        print("🎨 Vogue Image Scraper")
        print("=" * 60)
        print(f"📁 CSV source: {self.csv_path}")
        print(f"📂 Sortie 512x512: {self.output_512}")
        print(f"📂 Sortie 224x224: {self.output_224}")
        print(f"🎯 Maximum d'images: {self.max_images}")
        print("=" * 60)
        print()
        
        # Lire le CSV
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            
            print("⏳ Téléchargement en cours...\n")
            
            for row in reader:
                # Arrêter si on a atteint le maximum
                if self.stats['successful'] >= self.max_images:
                    print(f"\n✅ Maximum de {self.max_images} images atteint!")
                    break
                
                self.stats['total_processed'] += 1
                
                # Extraire les informations
                designer = row.get('designer', 'unknown')
                saison = row.get('saison', 'unknown')
                annee = row.get('annee', 'unknown')
                url = row.get('photo', '')
                num = row.get('num', '0')
                
                if not url:
                    continue
                
                # Générer le nom de fichier
                filename = self.generate_filename(designer, saison, annee, num)
                
                # Télécharger et redimensionner
                success = self.download_and_resize(url, filename)
                
                # Afficher la progression tous les 100 téléchargements
                if self.stats['successful'] % 100 == 0 and self.stats['successful'] > 0:
                    print(f"📥 {self.stats['successful']}/{self.max_images} images téléchargées...")
                
                # Pause pour éviter de surcharger le serveur
                time.sleep(0.1)
        
        # Afficher les statistiques finales
        self.print_stats()
    
    def print_stats(self):
        """Affiche les statistiques finales"""
        print("\n")
        print("=" * 60)
        print("📊 STATISTIQUES FINALES")
        print("=" * 60)
        print(f"📋 Lignes CSV traitées:  {self.stats['total_processed']}")
        print(f"✅ Images téléchargées:  {self.stats['successful']}")
        print(f"⏭️  Images déjà présentes: {self.stats['skipped']}")
        print(f"❌ Échecs:               {self.stats['failed']}")
        print("=" * 60)
        print(f"📂 Images 512x512: {self.output_512}")
        print(f"📂 Images 224x224: {self.output_224}")
        print("=" * 60)


def main():
    """Point d'entrée principal"""
    
    # Configuration
    CSV_PATH = r"C:\Users\adrib\Documents\ESILV\A5\CV and Deep Learning\Project\photo_base_saison.csv"
    OUTPUT_BASE = "/home/claude/data"
    MAX_IMAGES = 5000
    
    # Vérifier que le CSV existe
    if not os.path.exists(CSV_PATH):
        print(f"❌ Erreur: Le fichier CSV '{CSV_PATH}' n'existe pas!")
        sys.exit(1)
    
    # Créer et lancer le scraper
    scraper = VogueImageScraper(
        csv_path=CSV_PATH,
        output_base=OUTPUT_BASE,
        max_images=MAX_IMAGES
    )
    
    try:
        scraper.scrape()
        print("\n✨ Scraping terminé avec succès!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Scraping interrompu par l'utilisateur")
        scraper.print_stats()
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
