from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, schema_of_json
from pyspark.sql.types import StringType, IntegerType, DoubleType, ArrayType, StructType
import pyspark.sql.functions as F
import os
from pathlib import Path
import json
import os
import shutil

#import src
from src.utils.config import Config

class DataLoaderExploration:
    """Chargement et exploration des données arXiv"""
    
    def __init__(self, spark_session=None, config=Config):
            if spark_session:
                self.spark = spark_session
            else:
                self.spark = self._create_spark_session()
            self.config = config
    
    def _create_spark_session(self):
        """Crée une session Spark connectée au cluster existant"""
        return SparkSession.builder \
            .appName("arXiv-DataLoader")\
            .master("spark://tawfekh-d:7077")\
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.driver.bindAddress", "127.0.0.1") \
            .getOrCreate()

    
    # def _create_spark_session(self):
    #     """Crée une session Spark par défaut"""
    #     return SparkSession.builder \
    #         .appName("arXiv-DataLoader") \
    #         .getOrCreate()
    
    def load_json_data(self, file_path,sample_size=None):
        """Charger JSON arXiv"""
        print(f"Chargement des données depuis {file_path}...")
        #df = self.spark.read.option("multiline", "true").json(file_path)
        df = self.spark.read.json(file_path)
         # Échantillonnage si spécifié
        if sample_size and sample_size < df.count():
            fraction = sample_size / df.count()
            df = df.sample(fraction=fraction, seed=42)
            print(f" Échantillon de {sample_size} articles chargé")
        else:
            print(f" {df.count():,} articles chargés")
        
        return df
    
    def load_csv_sample(self, file_path):
        """Charger un CSV de test"""
        df = self.spark.read.csv(file_path, header=True, inferSchema=True)
        return df
    
    def explore_data(self,df, num_rows=5):
        """Exploration basique des données"""
        print("\n =Exploration basique des données ===")
        print(f"Nombre de lignes: {df.count()}")
        print(f"Nombre de colonnes: {len(df.columns)}")
        print(f"\nSchéma:")
        df.printSchema()
        print(f"\nPremières {num_rows} lignes:")
        df.show(num_rows, truncate=50)
    def statistic_data(self,df):
        # Statistiques
        print(f"\n_Statistiques des colonnes numériques:")
        df.describe().show(truncate=False)
        #  Analyse par colonne
        print("\nAnalyse par colonne :")
        
        for i, column in enumerate(df.columns, 1):
            print(f"\n[{i}] {column} :")
            
            # Type de données
            dtype = df.schema[column].dataType
            print(f"   Type : {dtype}")
        
            # Pourcentage de valeurs nulles
            total = df.count()
            nulls = df.filter(col(column).isNull()).count()
            # nas = df.filter(col(column).isNa()).count()
            if nulls > 0:
                print(f"    Valeurs nulles : {nulls}/{total} ({nulls/total*100:.1f}%)")
            
            # Valeurs distinctes pour les chaînes
            if isinstance(dtype, StringType):
                distinct = df.select(column).distinct().count()
                print(f"   Valeurs distinctes : {distinct}")
                
                if distinct <= 20 and distinct > 0:
                    print(f"   Exemples :")
                    df.select(column).distinct().limit(5).show(truncate=50)
        
        return df
    
    def reduce_datset(self,df,essentials_cols):
        """Réduit le dataset aux colonnes essentielles"""
        
        # Vérifier quelles colonnes existent
        available_cols = [col for col in essentials_cols if col in df.columns]
        missing_cols = [col for col in essentials_cols if col not in df.columns]
        
        if missing_cols:
            print(f"  Colonnes manquantes : {missing_cols}")
        
        df_selected = df.select(*available_cols)
        print("\n=== APRÈS RÉDUCTION AUX COLONNES IMPORTANTES ===")
        self.explore_data(df_selected)
        return df_selected
  
    def save_as_single_json(self,df, output_file):
        """
        Sauvegarde un DataFrame Spark en un seul fichier JSON
        """
        
        print(f"Sauvegarde en cours vers {output_file}...")
        
        # Créer un dossier temporaire
        temp_dir = "temp_spark_output"
        
        # Écrire le DataFrame en une seule partition
        df.coalesce(1).write.mode("overwrite").json(temp_dir)
        
        # Trouver le fichier JSON généré
        for file in os.listdir(temp_dir):
            if file.startswith("part-") and file.endswith(".json"):
                source_path = os.path.join(temp_dir, file)
                break
        
        # Copier et renommer
        shutil.copy(source_path, output_file) # type: ignore
        
        # Nettoyer le dossier temporaire
        shutil.rmtree(temp_dir)
        
        print(f"Fichier sauvegardé : {output_file}")
        print(f" Taille : {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        return output_file


    def distribution_variables(self,df):
        """Analyse des variables du dataset"""

        print(" ANALYSE DES categories")
        # Analyse des catégories
        categories_df = df.select("categories")
        categories_df = categories_df.withColumn(
            "category_list", 
            F.split(F.col("categories"), " ")
        )
        
        # Compter les catégories
        categories_exploded = categories_df.select(
            F.explode("category_list").alias("category")
        )
        category_counts = categories_exploded.groupBy("category") \
            .count() \
            .orderBy(F.col("count").desc())
        #--------------------
        print("\n Distribution des catégories:")
        for row in category_counts:
            print(f"  {row['category']}: {row['count']}")
        
        # Distribution des catégories par article
        categories_per_article = categories_df.withColumn(
            "num_categories", 
            F.size("category_list")
        )
        
        categories_per_article.select(
            F.mean("num_categories").alias("moyenne"),
            F.stddev("num_categories").alias("ecart_type"),
            F.min("num_categories").alias("min"),
            F.max("num_categories").alias("max")
        ).show()

        print("\n" + "="*30)
        print(" ANALYSE DES TITRES")
        
        # 1. Longueur des titres
        df_with_lengths = df.withColumn("title_length_chars", F.length(F.col("title"))) \
                        .withColumn("title_length_words", F.size(F.split(F.col("title"), " ")))
        
        print("\n Statistiques de longueur des titres:")
        df_with_lengths.select(
            F.mean("title_length_chars").alias("Longueur moyenne (caractères)"),
            F.mean("title_length_words").alias("Mots moyens par titre"),
            F.stddev("title_length_chars").alias("Écart-type caractères"),
            F.stddev("title_length_words").alias("Écart-type mots")
        ).show(truncate=False)
        
        print("\n" + "="*30)
        print(" ANALYSE DES ABSTRACTS")
    
        
        # 1. Longueur des abstracts
        df_with_abstract_lengths = df.withColumn("abstract_length_chars", F.length(F.col("abstract"))) \
                                    .withColumn("abstract_length_words", F.size(F.split(F.col("abstract"), " ")))
        
        print("\n Statistiques de longueur des abstracts:")
        abstract_stats = df_with_abstract_lengths.select(
            F.mean("abstract_length_chars").alias("Longueur moyenne (caractères)"),
            F.mean("abstract_length_words").alias("Mots moyens par abstract"),
            F.min("abstract_length_chars").alias("Longueur minimale"),
            F.max("abstract_length_chars").alias("Longueur maximale"),
            F.stddev("abstract_length_chars").alias("Écart-type")
        ).collect()[0]

    def create_sample_dataset(self, df, output_path, sample_size=10000):
        """Crée un échantillon pour les tests"""
        print(f"Création d'un échantillon de {sample_size} articles...")
        
        sample_df = df.limit(sample_size)
        self.save_as_single_json(sample_df, output_path)
        
        return sample_df