import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pyspark.sql.functions as F
import warnings
def distribution_categories(df):
    
    # Analyser la distribution des labels
   
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
    #-------------------------------
    # Visualisation
    category_counts_pd = category_counts.limit(40).toPandas()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='count', y='category', data=category_counts_pd)
    plt.title('Top 40 des catégories les plus fréquentes')
    plt.tight_layout()
    plt.savefig('../data/results/top_categories.png')
    plt.show()
    
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
# distribution_categories(df)
def analyze_titles(df):
    """Analyse approfondie des titres"""
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
    
    # 2. Distribution des longueurs
    length_stats_pd = df_with_lengths.select(
        "title_length_chars", "title_length_words"
    ).sample(False, 0.01).toPandas()  # Échantillon pour visualisation
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Histogramme des caractères
    axes[0].hist(length_stats_pd['title_length_chars'], bins=50, alpha=0.7, color='skyblue')
    axes[0].axvline(length_stats_pd['title_length_chars'].mean(), color='red', 
                   linestyle='dashed', linewidth=2, label=f'Moyenne: {length_stats_pd["title_length_chars"].mean():.0f}')
    axes[0].set_xlabel('Nombre de caractères')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution de la longueur des titres (caractères)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Histogramme des mots
    axes[1].hist(length_stats_pd['title_length_words'], bins=30, alpha=0.7, color='lightgreen')
    axes[1].axvline(length_stats_pd['title_length_words'].mean(), color='red', 
                   linestyle='dashed', linewidth=2, label=f'Moyenne: {length_stats_pd["title_length_words"].mean():.1f}')
    axes[1].set_xlabel('Nombre de mots')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution de la longueur des titres (mots)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/results/title_length_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    return df_with_lengths

def analyze_abstracts(df):
    """Analyse approfondie des abstracts"""
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
    
    print(f" Longueur moyenne : {abstract_stats['Longueur moyenne (caractères)']:.0f} caractères")
    print(f" Mots moyens : {abstract_stats['Mots moyens par abstract']:.0f} mots")
    print(f" Min/Max : {abstract_stats['Longueur minimale']} / {abstract_stats['Longueur maximale']} caractères")
    
    # 2. Distribution des longueurs (échantillon pour visualisation)
    abstract_sample = df_with_abstract_lengths.select(
        "abstract_length_chars", "abstract_length_words"
    ).sample(False, 0.005).toPandas()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogramme des caractères
    axes[0].hist(abstract_sample['abstract_length_chars'], bins=50, alpha=0.7, color='salmon')
    axes[0].axvline(abstract_sample['abstract_length_chars'].mean(), color='red', 
                   linestyle='dashed', linewidth=2, 
                   label=f'Moyenne: {abstract_sample["abstract_length_chars"].mean():.0f}')
    axes[0].set_xlabel('Nombre de caractères')
    axes[0].set_ylabel('Fréquence')
    axes[0].set_title('Distribution de la longueur des abstracts (caractères)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot des mots
    axes[1].boxplot(abstract_sample['abstract_length_words'], vert=False)
    axes[1].set_xlabel('Nombre de mots')
    axes[1].set_title('Distribution de la longueur des abstracts (mots)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/results/abstract_length_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

    return df_with_abstract_lengths