#!/usr/bin/env python3
"""
Script simplifié d'entraînement des modèles
"""

import sys
import os
import argparse
import traceback
import json
import numpy as np
from notebooks import (
    DataLoaderExploration, Config, DataPreprocessor, FeatureEngineer,
    ClassifierChainsModel, BinaryRelevanceModel, ModelEvaluator, MultiLabelMetrics
)
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser(description='Entraînement des modèles arXiv')
    parser.add_argument('--input', type=str, default='data/processed/data_with_features.parquet',
                       help='Chemin des données avec features')
    parser.add_argument('--top-n', type=int, default=30,
                       help='Nombre de catégories')
    parser.add_argument('--skip-br', action='store_true',
                       help='Sauter Binary Relevance')
    parser.add_argument('--skip-cc', action='store_true',
                       help='Sauter Classifier Chains')

    args = parser.parse_args()

    print("="*60)
    print(" ENTRAÎNEMENT DES MODÈLES")
    print("="*60)

    try:
        # Configuration
        config = Config()
        models_path = config.get_data_config()['models_path']

        # Initialisation
        loader = DataLoaderExploration()
        preprocessor = DataPreprocessor(loader.spark)
        feature_engineer = FeatureEngineer(loader.spark)
        evaluator = ModelEvaluator(loader.spark)  # Instanciation correcte

        # Chargement des données
        print(f" Chargement: {args.input}")
        if not os.path.exists(args.input):
            print(f" Fichier non trouvé: {args.input}")
            return

        df = loader.spark.read.parquet(args.input)
        print(f" {df.count():,} articles chargés")

        # Vérifier si les features existent déjà
        if 'features' not in df.columns:
            print(" Création des features...")
            df = feature_engineer.run_full_feature_engineering(df)

        # Préparer les labels
        br_model = BinaryRelevanceModel(loader.spark)
        df_with_labels = br_model.prepare_labels(df, top_n=args.top_n)
        label_columns = br_model.label_columns
        print(f" {len(label_columns)} labels préparés")

        # Split des données
        train_df, val_df, test_df = preprocessor.create_train_test_split(
            df_with_labels, test_size=0.2, validation_size=0.1
        )
        train_df.cache()
        test_df.cache()

        # Résultats
        all_results = {}

        # 1. Binary Relevance
        if not args.skip_br:
            print("\n ENTRAÎNEMENT BINARY RELEVANCE")
            br_model.train(train_df, features_col="features")

            # Prédictions
            predictions_br = br_model.predict(test_df)

            # Sauvegarder
            br_output = os.path.join(models_path, "binary_relevance")
            br_model.save_models(br_output)
            print(f" Modèle sauvegardé: {br_output}")

            # Évaluation
            pred_cols_br = [f"{label}_pred" for label in label_columns]
            metrics_br = evaluator.evaluate_model(
                predictions_br,
                label_columns,
                pred_cols_br,
                model_name="Binary Relevance"
            )
            all_results["binary_relevance"] = metrics_br

        # 2. Classifier Chains
        if not args.skip_cc:
            print("\n ENTRAÎNEMENT CLASSIFIER CHAINS")

            # Préparer les données pour CC (les labels sont déjà préparés par br_model.prepare_labels)
            cc_model = ClassifierChainsModel(loader.spark)
            cc_model.train(train_df, label_columns, features_col="features")

            # Prédictions
            predictions_cc = cc_model.predict(test_df)

            # Sauvegarder
            cc_output = os.path.join(models_path, "classifier_chains")
            cc_model.save_models(cc_output)
            print(f" Modèle sauvegardé: {cc_output}")

            # Évaluation
            pred_cols_cc = [f"{label}_pred" for label in label_columns]
            metrics_cc = evaluator.evaluate_model(
                predictions_cc,
                label_columns,
                pred_cols_cc,
                model_name="Classifier Chains"
            )
            all_results["classifier_chains"] = metrics_cc

        # Nettoyage
        train_df.unpersist()
        test_df.unpersist()

        # Sauvegarder les résultats
        if all_results:
            results_path = os.path.join("data", "results", "training_results.json")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)

            with open(results_path, 'w') as f:
                json.dump(all_results, f, indent=4)

            print(f"\n Résultats sauvegardés: {results_path}")

            # Comparer les modèles
            evaluator.compare_models(list(all_results.values()))

        print("\n ENTRAÎNEMENT TERMINÉ")
        loader.spark.stop()

    except Exception as e:
        print(f"\n ERREUR: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
