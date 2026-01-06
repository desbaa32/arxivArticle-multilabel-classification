"""
Script de feature engineering pour arXiv
Objectif: Créer les features (TF-IDF, statistiques, etc.) à partir des données prétraitées
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pathlib import Path

# Importations depuis votre structure
from scripts import Config,DataPreprocessor,DataLoaderExploration,FeatureEngineer
import pyspark.sql.functions as F

def main():
    """Script principal de feature engineering"""
    
    # Parser les arguments
    parser = argparse.ArgumentParser(description='Feature engineering pour arXiv')
    parser.add_argument('--input-data', type=str, required=True,
                       help='Chemin vers les données prétraitées')
    parser.add_argument('--text-col', type=str, default='combined_text',
                       help='Colonne de texte à utiliser (défaut: combined_text)')
    parser.add_argument('--vocab-size', type=int, default=5000,
                       help='Taille du vocabulaire TF-IDF (défaut: 5000)')
    
    args = parser.parse_args()
    
    print("="*20)
    print(" SCRIPT DE FEATURE ENGINEERING ARXIV")
    
    try:
        # 1. Charger la configuration
        print("\n Étape 1: Chargement de la configuration")
        config = Config()
        data_config = config.get_data_config()
        # 2. Initialiser les composants
        print(" Étape 2: Initialisation des composants")
        loader = DataLoaderExploration()
        
        # Configurer le feature engineer avec les paramètres
        feature_engineer = FeatureEngineer(
            loader.spark, 
            vocab_size=args.vocab_size
        )
        
        # 3. Charger les données prétraitées
        print(f"\n Étape 3: Chargement des données: {args.input_data}")
        
        if not os.path.exists(args.input_data):
            print(f" Fichier non trouvé: {args.input_data}")
            print(" Exécutez d'abord: python scripts/run_preprocessing.py")
            sys.exit(1)
        
        # Détecter le format et charger
        if args.input_data.endswith('.json'):
            df = loader.spark.read.json(args.input_data)
        else:
            print(f" Format non supporté: {args.input_data}")
            print(" Utilisez .parquet ou .json")
            sys.exit(1)
        
        print(f" Données chargées: {df.count():,} articles, {len(df.columns)} colonnes")
        
        # 5. Feature Engineering
        print("\n  Étape 4: Feature Engineering")
        print(f"   - Colonne texte: {args.text_col}")
        print(f"   - Taille vocabulaire: {args.vocab_size}")
        print(f"   - Utilisation de N-grammes")
        
        
        # Exécuter le feature engineering
        df_features = feature_engineer.run_full_feature_engineering(
            df, 
            text_col=args.text_col
        )
        
        
        # 6. Analyser les features créées
        print("\n Étape 5: Analyse des features")
        
        # Compter le nombre de features créées
        feature_columns = [col for col in df_features.columns if 'feature' in col.lower()]
        other_columns = [col for col in df_features.columns if 'feature' not in col.lower()]
        
        print(f" Features créées: {len(feature_columns)} colonnes de features")
        print(f"   Colonnes de features: {', '.join(feature_columns[:5])}...")
        print(f"   Autres colonnes: {len(other_columns)} (id, texte, labels, etc.)")
        
        # Afficher la dimension des features principales
        if 'features' in df_features.columns:
            # Échantillonner un vecteur pour voir sa dimension
            sample = df_features.select('features').limit(1).collect()[0]['features']
            print(f"   Dimension du vecteur 'features': {len(sample)}")
        
        # Statistiques sur les textes
        if args.text_col in df_features.columns:
            stats = df_features.select(
                F.mean(F.length(F.col(args.text_col))).alias("longueur_moyenne"),
                F.stddev(F.length(F.col(args.text_col))).alias("ecart_type"),
                F.count("*").alias("total")
            ).collect()[0]
            
            print(f"\n Statistiques texte:")
            print(f"   Longueur moyenne: {stats['longueur_moyenne']:.0f} caractères")
            print(f"   Écart-type: {stats['ecart_type']:.0f} caractères")
            print(f"   Total articles: {stats['total']:,}")
        
        # 7. Sauvegarder les données avec features
        
        print("\n Étape 6: Sauvegarde des données prétraitées")
        output_name = "features_arxiv_data.json"
        # Chemin de sortie
        output_path = os.path.join(data_config['processed_path'], output_name)
        # Sauvegarder en parquet (format optimal pour Spark)
        loader.save_as_single_json(df_features, output_path)
        print(f"\nFEATURE ENGINEERING TERMINÉ AVEC SUCCÈS")
        print(f" Données sauvegardées: {output_path}")
        print(f"Statistiques finales:")
        print(f"   - Articles: {df_features.count():,}")
        print(f"   - Colonnes totales: {len(df_features.columns)}")
        print(f"   - Colonnes de features: {len(feature_columns)}")
        
        # Afficher un aperçu des données
        print(f"\n Aperçu des données (premières colonnes):")
        df_features.select(df_features.columns[:5]).show(3, truncate=30)
        output_name = "sampled_features_arxiv_data.json"
        # 8. Option: créer un échantillon pour tests rapides
        output_path = os.path.join(data_config['processed_path'], output_name)
        df_sample = df_features.limit(10000)
        loader.save_as_single_json(df_sample, output_path)
        print(f"\n Échantillon créé: {output_path} (10,000 articles)")
        # 9. Option: exporter les métadonnées des features
        metadata = {
            'input_data': args.input_data,
            'text_column': args.text_col,
            'vocab_size': args.vocab_size,
            'use_ngrams': args.use_ngrams,
            'use_word2vec': args.use_word2vec,
            'output_file': output_path,
            'total_samples': df_features.count(),
            'feature_columns': feature_columns,
            'total_columns': len(df_features.columns),
            # 'timestamp': str(pd.Timestamp.now())
        }
        
        meta_name = "sampled_features_arxiv_data_metadata.json"
        metadata_path = os.path.join(data_config['processed_path'], meta_name)
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f" Métadonnées sauvegardées: {metadata_path}")
        loader.spark.stop()
        
    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()