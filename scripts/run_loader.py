#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de chargement des données arXiv
Objectif: Charger et explorer les données avant le préprocessing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

# from src.utils.config import Config
# from loaderExploration import DataLoaderExploration
from scripts import DataLoaderExploration,Config
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

import argparse

def main():
    """Script principal de chargement des données"""
    
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Charger les données arXiv')
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='Taille de l\'échantillon (défaut: 50000)')
    parser.add_argument('--full', action='store_true',
                       help='Charger le dataset complet')
    
    args = parser.parse_args()
    
    print("="*60)
    print(" SCRIPT DE CHARGEMENT DES DONNÉES ARXIV")
    print("="*60)
    
    try:
        # 1. Charger la configuration
        print("\nEtape 1: Chargement de la configuration")
        config = Config()
        spark_config = config.get_spark_config()
        data_config = config.get_data_config()
        
        # 2. Chemin des données
        raw_data_path = data_config.get('raw_path', 'data/raw/arxiv-metadata-oai-snapshot.json')
        project_root = Path(__file__).parent.parent
        full_data_path = os.path.join(project_root, raw_data_path)
        
        print(f"Dossier actuel: {project_root}")
        print(f"Chemin des données brutes: {full_data_path}")
        
        # 3. Vérifier si le fichier existe
        if not os.path.exists(full_data_path):
            print(f"Fichier non trouve: {full_data_path}")
            print("Solutions possibles:")
            print("  1. Telecharger le dataset depuis Kaggle:")
            print("     https://www.kaggle.com/datasets/Cornell-University/arxiv")
            sys.exit(1)
        
        # 4. Déterminer la taille de l'échantillon depuis les arguments
        if args.full:
            sample_size = None
            print("Mode: Chargement du dataset complet")
        else:
            sample_size = args.sample_size
            print(f"Mode: Echantillon de {sample_size} articles")
        
        # 5. Initialiser DataLoader
        print("\nEtape 2: Initialisation de DataLoader")
        loader = DataLoaderExploration()
        
        # 6. Charger les données
        print(f"\nEtape 3: Chargement des données")
        print("Cette operation peut prendre plusieurs minutes...")
        df = loader.load_json_data(full_data_path, sample_size=sample_size)
        
        # 7. Afficher les informations de base
        print("\nEtape 4: Exploration initiale")
        loader.explore_data(df)
        loader.statistic_data(df)
        
        # 8. Filtrer les colonnes essentielles
        print("\nEtape 5: Filtrage des colonnes essentielles")
        essential_cols = ['id', 'title', 'abstract', 'categories']
        df_filtered = loader.reduce_datset(df, essential_cols)
        
        # 9. Analyser
        print("\nEtape 6: Analyse")
        loader.distribution_variables(df_filtered)
        
        # 10. Sauvegarder les données filtrées
        print("\nEtape 7: Sauvegarde des donnees")
        
        if sample_size:
            output_name = f"arxiv_sample_{sample_size}_filtered.json"
        else:
            output_name = "arxiv_full_filtered.json"
        
        # Utiliser le chemin de la configuration pour le dossier processed
        output_path = os.path.join(data_config['processed_path'], output_name)
        
        loader.save_as_single_json(df_filtered, output_path)
        print(f"Donnees sauvegardees dans: {output_path}")
        
        loader.spark.stop()
        
    except Exception as e:
        print(f"\nErreur lors du chargement: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# # Charger 10,000 articles
# python scripts/run_loader.py --sample-size 10000

# # Charger 50,000 articles (défaut)
# python scripts/run_loader.py

# # Charger tout le dataset
# python scripts/run_loader.py --full