
"""
Script de preprocessing des données arXiv
Objectif: Nettoyer, préparer et transformer les données pour la modélisation
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pyspark.sql.functions as F
# Ajouter le répertoire parent au path
#sys.path.append(str(Path(__file__).parent.parent))
from scripts import Config,DataPreprocessor,DataLoaderExploration




def load_reduced_data(loader, config, data_source="processed"):
    """
    Charge les données réduites depuis le dossier processed
    ou charge et réduit les données brutes si nécessaire
    """
    data_config = config.get_data_config()
    
    if data_source == "processed":
        # Chercher le fichier réduit le plus récent dans processed
        processed_path = Path(data_config['processed_path'])
        json_files = list(processed_path.glob("*_filtered.json"))
        
        if json_files:
            # Prendre le fichier le plus récent
            latest_file = max(json_files, key=os.path.getctime)
            print(f"Chargement des données réduites: {latest_file}")
            return loader.spark.read.json(str(latest_file))
        else:
            print("  Aucun fichier réduit trouvé, chargement depuis raw...")
            data_source = "raw"
    
    if data_source == "raw":
        # Charger depuis raw et réduire
        raw_path = data_config['raw_path']
        full_raw_path = Path(__file__).parent.parent / raw_path
        
        print(f" Chargement des données brutes: {full_raw_path}")
        df = loader.spark.read.json(str(full_raw_path))
        
        # Réduire aux colonnes essentielles
        essential_cols = ['id', 'title', 'abstract', 'categories']
        available_cols = [col for col in essential_cols if col in df.columns]
        df_reduced = df.select(*available_cols)
        
        print(f" Données réduites: {len(available_cols)} colonnes")
        return df_reduced

def main():
    """Script principal de preprocessing"""
    
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Préprocessing des données arXiv')
    parser.add_argument('--source', type=str, choices=['raw', 'processed'], default='processed',
                       help='Source des données: raw (brutes) ou processed (réduites)')
    parser.add_argument('--sample-size', type=int, default=50000,
                       help='Taille de l\'échantillon (pour raw seulement)')
    parser.add_argument('--output-name', type=str, default='preprocessed_data',
                       help='Nom du fichier de sortie (sans extension)')
    
    args = parser.parse_args()
    
    print("="*20)
    print(" SCRIPT DE PRÉPROCESSING DES DONNÉES ARXIV")
    
    
    try:
        # 1. Charger la configuration
        print("\n Étape 1: Chargement de la configuration")
        config = Config()
        spark_config = config.get_spark_config()
        data_config = config.get_data_config()
        # 2. Initialiser DataLoader
        print(" Étape 2: Initialisation des composants")
        loader = DataLoaderExploration()
        preprocessor = DataPreprocessor(loader.spark)
        
        # 3. Charger les données
        print(f"\n Étape 3: Chargement des données depuis {args.source}")
        
        if args.source == 'raw' and args.sample_size:
            # Charger un échantillon depuis raw
            raw_path = config.get_data_config()['raw_path']
            full_raw_path = Path(__file__).parent.parent / raw_path
            df = loader.load_json_data(str(full_raw_path), sample_size=args.sample_size)
            
            # Filtrer les colonnes essentielles
            
            df = loader.reduce_datset(df, ['id', 'title', 'abstract', 'categories'])
        else:
            # Charger depuis processed
            df = load_reduced_data(loader, config, args.source)
        
        print(f" Données chargées....")
        
        # 4. Préprocessing basique
        print("\n Étape 4: Préprocessing basique")
        df = preprocessor.run_full_preprocessing(df)
        
        # 5. Statistiques après préprocessing
        print("\nÉtape 5: Statistiques post-préprocessing")
        
        # Nombre de catégories par article
        if "num_categories" in df.columns:
            print("\nDistribution du nombre de catégories par article:")
            df.groupBy("num_categories").count().orderBy("num_categories").show(10)
        
        
        # 6. Sauvegarder les données prétraitées
        print("\n Étape 6: Sauvegarde des données prétraitées")
        output_name = "processed_arxiv_data.json"
        # Chemin de sortie
        output_path = os.path.join(data_config['processed_path'], output_name)
        # Sauvegarder en parquet (format optimal pour Spark)
        loader.save_as_single_json(df, output_path)
        
        
        # # 7. Option: créer un échantillon pour tests
        # sample_output = output_dir / f"{args.output_name}_sample.parquet"
        # df_sample = df.limit(10000)
        # loader.save_data(df_sample, str(sample_output), format='parquet')
        # print(f"   - Échantillon créé: {sample_output} (10,000 articles)")
        loader.spark.stop()
    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    
if __name__ == "__main__":
    main()