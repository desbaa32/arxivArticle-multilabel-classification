# src/data/preprocessing.py
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import re

class DataPreprocessor:
    """Classe pour le préprocessing des données texte"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
    
    def clean_text_column(self, df, text_col, new_col=None):
        """
        Nettoie une colonne de texte
        """
        if new_col is None:
            new_col = f"clean_{text_col}"
        
        print(f" Nettoyage de la colonne: {text_col}")
        
        # Chaîne de nettoyage
        cleaned = F.lower(F.col(text_col))  # Minuscules
        
        # Supprimer les caractères spéciaux (garder lettres, chiffres, espaces)
        cleaned = F.regexp_replace(cleaned, r'[^a-zA-Z0-9\s]', ' ')
        
        # Supprimer les URLs
        cleaned = F.regexp_replace(cleaned, r'https?://\S+|www\.\S+', ' ')
        
        # Supprimer les nombres isolés
        cleaned = F.regexp_replace(cleaned, r'\b\d+\b', ' ')
        
        # Supprimer les espaces multiples
        cleaned = F.regexp_replace(cleaned, r'\s+', ' ')
        
        # Trim
        cleaned = F.trim(cleaned)
        
        return df.withColumn(new_col, cleaned)
    
    def combine_text_columns(self, df, columns, new_col="combined_text"):
        """
        Combine plusieurs colonnes texte en une seule
        """
        print(f" Combinaison des colonnes: {columns}")
        
        # Vérifier que les colonnes existent
        existing_cols = [col for col in columns if col in df.columns]
        
        if len(existing_cols) != len(columns):
            missing = set(columns) - set(existing_cols)
            print(f"  Colonnes manquantes: {missing}")
        
        # Combiner avec espace
        combined = F.concat_ws(" ", *[F.col(col) for col in existing_cols])
        
        return df.withColumn(new_col, combined)
    
    def process_categories(self, df, categories_col="categories"):
        """
        Traite la colonne des catégories
        """
        print(f" Traitement des catégories")
        
        # Séparer les catégories en liste
        df = df.withColumn("category_list", F.split(F.col(categories_col), " "))
        
        # Nombre de catégories par article
        df = df.withColumn("num_categories", F.size(F.col("category_list")))
        
        # Catégorie principale (première)
        df = df.withColumn("main_category", F.col("category_list")[0])
        
        # Extraire le domaine (ex: 'cs' de 'cs.AI')
        df = df.withColumn(
            "domain",
            F.split(F.col("main_category"), "\\.")[0]
        )
        
        return df
    
    def prepare_multi_label_targets(self, df, top_n=30, categories_col="categories"):
        """
        Prépare les targets pour la classification multi-label
        """
        print(f" Préparation des targets multi-label (top {top_n})")
        
        # Extraire toutes les catégories et compter les fréquences
        categories_df = df.select(F.explode(F.split(F.col(categories_col), " ")).alias("category"))
        category_counts = categories_df.groupBy("category").count().orderBy(F.desc("count"))
        
        # Prendre les top N catégories
        top_categories = [row['category'] for row in category_counts.limit(top_n).collect()]
        
        print(f" Top {len(top_categories)} catégories sélectionnées")
        
        # Créer une colonne binaire pour chaque catégorie
        for cat in top_categories:
            safe_name = cat.replace('.', '_').replace('-', '_')
            df = df.withColumn(
                f"label_{safe_name}",
                F.when(F.array_contains(F.split(F.col(categories_col), " "), cat), 1).otherwise(0)
            )
        
        # Créer une liste des colonnes de labels
        label_columns = [f"label_{cat.replace('.', '_').replace('-', '_')}" for cat in top_categories]
        
        # Créer un vecteur de labels
        from pyspark.ml.feature import VectorAssembler
        assembler = VectorAssembler(inputCols=label_columns, outputCol="label_vector")
        df = assembler.transform(df)
        
        return df, top_categories, label_columns
    
    def filter_by_text_length(self, df, text_col="combined_text", min_length=100):
        """
        Filtre les articles avec texte trop court
        """
        print(f" Filtrage par longueur de texte (min: {min_length} caractères)")
        
        initial_count = df.count()
        df_filtered = df.filter(F.length(F.col(text_col)) >= min_length)
        final_count = df_filtered.count()
        
        removed = initial_count - final_count
        print(f"{removed} articles supprimés, {final_count} restants")
        
        return df_filtered
    
    def remove_duplicates(self, df, id_col="id"):
        """
        Supprime les doublons basés sur l'ID
        """
        print(f" Suppression des doublons")
        
        initial_count = df.count()
        df_dedup = df.dropDuplicates([id_col])
        final_count = df_dedup.count()
        
        duplicates = initial_count - final_count
        if duplicates > 0:
            print(f" {duplicates} doublons supprimés")
        
        return df_dedup
    
    def create_train_test_split(self, df, test_size=0.2, validation_size=0.1, seed=42):
        """
        Crée les splits train/validation/test
        """
        print("  Division des données")
        
        # Split train/test
        train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=seed)
        
        # Split train/validation
        train_final, val_df = train_df.randomSplit(
            [1 - validation_size, validation_size], 
            seed=seed
        )
        
        print(f" Tailles des datasets:")
        print(f"  Train:      {train_final.count():,} échantillons")
        print(f"  Validation: {val_df.count():,} échantillons")
        print(f"  Test:       {test_df.count():,} échantillons")
        
        return train_final, val_df, test_df
    
    def run_full_preprocessing(self, df, text_columns=["title", "abstract"]):
        """
        Exécute le pipeline complet de préprocessing
        """
        print(" Démarrage du pipeline de préprocessing")
        
        # 1. Filtrer les colonnes essentielles
        essential_cols = ['id', 'title', 'abstract', 'categories']
        df = df.select([col for col in essential_cols if col in df.columns])
        
        # 2. Nettoyer les colonnes texte
        for col in text_columns:
            if col in df.columns:
                df = self.clean_text_column(df, col)
        
        # 3. Combiner les textes
        clean_cols = [f"clean_{col}" for col in text_columns if f"clean_{col}" in df.columns]
        if clean_cols:
            df = self.combine_text_columns(df, clean_cols, "combined_text")
        
        # 4. Traiter les catégories
        if "categories" in df.columns:
            df = self.process_categories(df)
        
        # 5. Filtrer par longueur
        if "combined_text" in df.columns:
            df = self.filter_by_text_length(df, "combined_text", min_length=100)
        
        # 6. Supprimer les doublons
        df = self.remove_duplicates(df)
        
        print(" Préprocessing terminé avec succès")
        
        return df