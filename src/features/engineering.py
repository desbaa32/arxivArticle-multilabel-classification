# src/features/engineering.py
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF,
    CountVectorizer, NGram, Word2Vec, VectorAssembler
)
from pyspark.ml import Pipeline
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

class FeatureEngineer:
    """Ingénierie des features pour le texte"""
    
    def __init__(self, spark_session, vocab_size=5000):
        self.spark = spark_session
        self.vocab_size = vocab_size
    
    def create_text_pipeline(self, input_col="combined_text"):
        """
        Crée un pipeline de traitement de texte
        """
        print(" Création du pipeline de traitement de texte")
        
        # Tokenizer
        tokenizer = Tokenizer(inputCol=input_col, outputCol="words")
        
        # StopWordsRemover
        remover = StopWordsRemover(
            inputCol="words", 
            outputCol="filtered_words",
            stopWords=StopWordsRemover.loadDefaultStopWords("english")
        )
        
        # HashingTF pour TF
        hashing_tf = HashingTF(
            inputCol="filtered_words",
            outputCol="raw_features",
            numFeatures=self.vocab_size
        )
        
        # IDF
        idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
        
        # Pipeline
        pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
        
        return pipeline
    
    def create_ngram_features(self, df, input_col="filtered_words", n=2):
        """
        Crée des features n-gram
        """
        print(f" Création des {n}-grammes")
        
        ngram = NGram(n=n, inputCol=input_col, outputCol=f"{n}_grams")
        
        # HashingTF pour les n-grammes
        ngram_tf = HashingTF(
            inputCol=f"{n}_grams",
            outputCol=f"raw_{n}gram_features",
            numFeatures=self.vocab_size // 2
        )
        
        # IDF
        ngram_idf = IDF(
            inputCol=f"raw_{n}gram_features",
            outputCol=f"{n}gram_tfidf"
        )
        
        # Appliquer
        df = ngram.transform(df)
        df = ngram_tf.transform(df)
        df = ngram_idf.fit(df).transform(df)
        
        return df
    
    def create_text_statistics(self, df, text_col="combined_text"):
        """
        Crée des statistiques textuelles basiques
        """
        print(" Calcul des statistiques textuelles")
        
        # Longueur en caractères
        df = df.withColumn("text_length", F.length(F.col(text_col)))
        
        # Nombre de mots
        df = df.withColumn("word_count", F.size(F.split(F.col(text_col), " ")))
        
        # Longueur moyenne des mots
        df = df.withColumn(
            "avg_word_length", 
            F.when(F.col("word_count") > 0, F.col("text_length") / F.col("word_count"))
             .otherwise(0)
        )
        
        # Mots uniques
        df = df.withColumn(
            "unique_words",
            F.size(F.array_distinct(F.split(F.col(text_col), " ")))
        )
        
        # Ratio de mots uniques
        df = df.withColumn(
            "unique_ratio",
            F.when(F.col("word_count") > 0, F.col("unique_words") / F.col("word_count"))
             .otherwise(0)
        )
        
        return df
    
    def create_metadata_features(self, df):
        """
        Crée des features à partir des métadonnées
        """
        print("Création des features de métadonnées")
        
        # Longueur du titre
        if "title" in df.columns:
            df = df.withColumn("title_length", F.length(F.col("title")))
        
        # Année de publication (si disponible)
        if "update_date" in df.columns:
            # Extraire l'année
            df = df.withColumn("year", F.year(F.to_date(F.col("update_date"))))
            
            # Catégoriser par période
            df = df.withColumn(
                "period",
                F.when(F.col("year") < 2000, "avant_2000")
                 .when(F.col("year") < 2010, "2000_2009")
                 .when(F.col("year") < 2015, "2010_2014")
                 .otherwise("apres_2015")
            )
        
        return df
    
    def combine_features(self, df, feature_columns=None):
        """
        Combine toutes les features en un vecteur
        """
        print(" Combinaison des features")
        
        # Colonnes par défaut
        if feature_columns is None:
            feature_columns = [
                "tfidf_features",
                "text_length",
                "word_count",
                "avg_word_length",
                "unique_ratio"
            ]
        
        # Garder seulement les colonnes qui existent
        existing_columns = [col for col in feature_columns if col in df.columns]
        
        print(f" Features à combiner: {existing_columns}")
        
        # Assembler
        assembler = VectorAssembler(
            inputCols=existing_columns,
            outputCol="features",
            handleInvalid="skip"
        )
        
        df_features = assembler.transform(df)
        
        print(f"Vecteur de features créé ({len(existing_columns)} dimensions)")
        
        return df_features
    
    def run_full_feature_engineering(self, df, text_col="combined_text"):
        """
        Exécute le pipeline complet d'ingénierie des features
        """
        print(" Démarrage du feature engineering")
        
        # 1. Pipeline de base (tokenization + TF-IDF)
        pipeline = self.create_text_pipeline(text_col)
        model = pipeline.fit(df)
        df = model.transform(df)
        
        # 2. N-grammes (bigrammes)
        df = self.create_ngram_features(df, "filtered_words", n=2)
        
        # 3. Statistiques textuelles
        df = self.create_text_statistics(df, text_col)
        
        # 4. Features de métadonnées
        df = self.create_metadata_features(df)
        
        # 5. Combiner toutes les features
        df = self.combine_features(df)
        
        print(" Feature engineering terminé")
        
        return df