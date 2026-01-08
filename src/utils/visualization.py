# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pyspark.sql.functions as F
import warnings
from pathlib import Path
import json
import os
warnings.filterwarnings('ignore')

class DataVisualizer:

    """Visualisations pour le projet arXiv"""
    
    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def distribution_categories(self, df):
        """Analyse la distribution des catégories arXiv"""
        
        print(" Analyse des catégories...")
        
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
        
        print("\n Distribution des catégories (top 10):")
        for row in category_counts.limit(10):
            print(f"  {row['category']}: {row['count']}")
        
        # Visualisation
        category_counts_pd = category_counts.limit(40).toPandas()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='count', y='category', data=category_counts_pd)
        plt.title('Top 40 des catégories les plus fréquentes')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, "top_categories.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Graphique sauvegardé: {output_path}")
        plt.show()
        
        # Distribution des catégories par article
        categories_per_article = categories_df.withColumn(
            "num_categories", 
            F.size("category_list")
        )
        
        print("\n Statistiques du nombre de catégories par article:")
        categories_per_article.select(
            F.mean("num_categories").alias("moyenne"),
            F.stddev("num_categories").alias("ecart_type"),
            F.min("num_categories").alias("min"),
            F.max("num_categories").alias("max")
        ).show()
        
        return category_counts
    
    def analyze_titles(self, df):
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
        ).sample(False, 0.01).toPandas()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogramme des caractères
        axes[0].hist(length_stats_pd['title_length_chars'], bins=50, alpha=0.7, color='skyblue')
        axes[0].axvline(length_stats_pd['title_length_chars'].mean(), color='red', 
                       linestyle='dashed', linewidth=2, 
                       label=f'Moyenne: {length_stats_pd["title_length_chars"].mean():.0f}')
        axes[0].set_xlabel('Nombre de caractères')
        axes[0].set_ylabel('Fréquence')
        axes[0].set_title('Distribution de la longueur des titres (caractères)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogramme des mots
        axes[1].hist(length_stats_pd['title_length_words'], bins=30, alpha=0.7, color='lightgreen')
        axes[1].axvline(length_stats_pd['title_length_words'].mean(), color='red', 
                       linestyle='dashed', linewidth=2, 
                       label=f'Moyenne: {length_stats_pd["title_length_words"].mean():.1f}')
        axes[1].set_xlabel('Nombre de mots')
        axes[1].set_ylabel('Fréquence')
        axes[1].set_title('Distribution de la longueur des titres (mots)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "title_length_distribution.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Graphique sauvegardé: {output_path}")
        plt.show()
        
        return df_with_lengths
    
    def analyze_abstracts(self, df):
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
        
        # 2. Distribution des longueurs
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
        output_path = os.path.join(self.output_dir, "abstract_length_distribution.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Graphique sauvegardé: {output_path}")
        plt.show()
        
        return df_with_abstract_lengths
    
    def plot_model_comparison(self, model_results: Dict[str, Dict], filename: str = "model_comparison.png"):
        """Compare visuellement les performances des modèles"""
        if not model_results:
            print("  Aucun résultat à visualiser")
            return
        
        model_names = list(model_results.keys())
        metrics = ['hamming_loss', 'subset_accuracy', 'micro_f1', 'macro_f1']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [model_results[name].get(metric, 0) for name in model_names]
            
            # Couleurs selon la performance
            colors = []
            for val in values:
                if metric == 'hamming_loss':
                    # Pour hamming_loss, plus bas = meilleur
                    if val == min(values):
                        colors.append('green')
                    else:
                        colors.append('red')
                else:
                    # Pour les autres, plus haut = meilleur
                    if val == max(values):
                        colors.append('green')
                    else:
                        colors.append('red')
            
            bars = ax.bar(model_names, values, color=colors, alpha=0.7)
            ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_ylabel('Score')
            ax.grid(axis='y', alpha=0.3)
            
            # Ajouter les valeurs
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Comparaison des Modèles', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Comparaison sauvegardée: {output_path}")
        plt.show()
    
    def plot_confusion_heatmap(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              label_names: Optional[List[str]] = None,  
                              filename: str = "confusion_heatmap.png"):
        """Crée une heatmap de la matrice de confusion"""
        n_labels = y_true.shape[1]
        
        # Calculer la matrice de confusion
        confusion = np.zeros((n_labels, n_labels))
        for i in range(n_labels):
            for j in range(n_labels):
                confusion[i, j] = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 1))
        
        # Normaliser par ligne
        confusion = confusion / confusion.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(12, 10))
        
        # Utiliser des noms de labels courts si fournis
        if label_names and len(label_names) == n_labels:
            tick_labels = [name.split('_')[-1] for name in label_names]
        else:
            tick_labels = [f"Label {i}" for i in range(n_labels)]
        
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=tick_labels, yticklabels=tick_labels)
        
        plt.title('Matrice de Confusion Normalisée', fontsize=14)
        plt.xlabel('Prédictions')
        plt.ylabel('Vérité terrain')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f" Heatmap sauvegardée: {output_path}")
        plt.show()