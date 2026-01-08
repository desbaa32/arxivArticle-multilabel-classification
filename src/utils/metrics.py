# src/utils/metrics.py
import numpy as np
from typing import List, Dict, Tuple
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

class MultiLabelMetrics:
    """Métriques pour l'évaluation multi-label"""
    
    @staticmethod
    def calculate_hamming_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule le Hamming Loss
        
        Args:
            y_true: Array numpy de shape (n_samples, n_labels) avec valeurs 0/1
            y_pred: Array numpy de shape (n_samples, n_labels) avec valeurs 0/1
            
        Returns:
            Hamming loss (plus bas est mieux)
        """
        n_samples, n_labels = y_true.shape
        loss = np.sum(y_true != y_pred) / (n_samples * n_labels)
        return float(loss)
    
    @staticmethod
    def calculate_subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcule la Subset Accuracy (exact match)
        
        Args:
            y_true: Array numpy de shape (n_samples, n_labels)
            y_pred: Array numpy de shape (n_samples, n_labels)
            
        Returns:
            Proportion d'échantillons parfaitement classés
        """
        matches = np.all(y_true == y_pred, axis=1)
        accuracy = np.mean(matches)
        return float(accuracy)
    
    @staticmethod
    def calculate_example_based_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcule les métriques par exemple
        
        Returns:
            Dict avec precision, recall, f1 par exemple
        """
        n_samples = y_true.shape[0]
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(n_samples):
            true_pos = np.sum(y_true[i] & y_pred[i])
            pred_pos = np.sum(y_pred[i])
            actual_pos = np.sum(y_true[i])
            
            precision = true_pos / pred_pos if pred_pos > 0 else 0
            recall = true_pos / actual_pos if actual_pos > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        return {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1_scores)
        }
    
    @staticmethod
    def calculate_label_based_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcule les métriques par label
        
        Returns:
            Dict avec macro et micro averages
        """
        n_labels = y_true.shape[1]
        
        # Métriques par label
        label_precisions = []
        label_recalls = []
        label_f1_scores = []
        
        for j in range(n_labels):
            true_pos = np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 1))
            false_pos = np.sum((y_true[:, j] == 0) & (y_pred[:, j] == 1))
            false_neg = np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 0))
            
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            
            label_precisions.append(precision)
            label_recalls.append(recall)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            label_f1_scores.append(f1)
        
        # Macro averages
        macro_precision = np.mean(label_precisions)
        macro_recall = np.mean(label_recalls)
        macro_f1 = np.mean(label_f1_scores)
        
        # Micro averages
        micro_true_pos = np.sum(y_true & y_pred)
        micro_false_pos = np.sum(~y_true & y_pred)
        micro_false_neg = np.sum(y_true & ~y_pred)
        
        micro_precision = micro_true_pos / (micro_true_pos + micro_false_pos) if (micro_true_pos + micro_false_pos) > 0 else 0
        micro_recall = micro_true_pos / (micro_true_pos + micro_false_neg) if (micro_true_pos + micro_false_neg) > 0 else 0
        
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        return {
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1
        }
    
    @staticmethod
    def extract_labels_from_spark(df: DataFrame, label_columns: List[str]) -> np.ndarray:
        """
        Extrait les labels d'un DataFrame Spark
        
        Args:
            df: DataFrame Spark
            label_columns: Liste des colonnes de labels
            
        Returns:
            Array numpy des labels
        """
        labels_data = df.select(*label_columns).collect()
        y = np.array([list(row) for row in labels_data], dtype=int)
        return y
    
    @staticmethod
    def evaluate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "") -> Dict:
        """
        Calcule toutes les métriques pour un modèle
        
        Returns:
            Dictionnaire avec toutes les métriques
        """
        metrics = {
            'model': model_name,
            'hamming_loss': MultiLabelMetrics.calculate_hamming_loss(y_true, y_pred),
            'subset_accuracy': MultiLabelMetrics.calculate_subset_accuracy(y_true, y_pred)
        }
        
        # Métriques par exemple
        example_metrics = MultiLabelMetrics.calculate_example_based_metrics(y_true, y_pred)
        metrics.update({
            'example_precision': example_metrics['precision'],
            'example_recall': example_metrics['recall'],
            'example_f1': example_metrics['f1']
        })
        
        # Métriques par label
        label_metrics = MultiLabelMetrics.calculate_label_based_metrics(y_true, y_pred)
        metrics.update(label_metrics)
        
        # Statistiques additionnelles
        metrics['avg_true_labels'] = float(np.mean(np.sum(y_true, axis=1)))
        metrics['avg_pred_labels'] = float(np.mean(np.sum(y_pred, axis=1)))
        
        return metrics


class ModelEvaluator:
    """Évaluateur de modèles avec Spark"""
    
    def __init__(self, spark_session):
        self.spark = spark_session
        self.metrics_calculator = MultiLabelMetrics()
    
    def evaluate_model(self, predictions_df: DataFrame, 
                      true_label_columns: List[str], 
                      pred_label_columns: List[str],
                      model_name: str = "") -> Dict:
        """
        Évalue un modèle à partir d'un DataFrame de prédictions
        
        Args:
            predictions_df: DataFrame Spark avec vraies et prédites
            true_label_columns: Colonnes des vraies labels
            pred_label_columns: Colonnes des labels prédits
            model_name: Nom du modèle pour le rapport
            
        Returns:
            Dictionnaire avec toutes les métriques
        """
        print(f"Évaluation du modèle: {model_name}")
        
        # Extraire les labels
        y_true = MultiLabelMetrics.extract_labels_from_spark(predictions_df, true_label_columns)
        y_pred = MultiLabelMetrics.extract_labels_from_spark(predictions_df, pred_label_columns)
        
        # Calculer toutes les métriques
        metrics = MultiLabelMetrics.evaluate_all_metrics(y_true, y_pred, model_name)
        
        # Afficher les résultats
        self._print_metrics_report(metrics)
        
        return metrics
    
    def compare_models(self, metrics_list: List[Dict]) -> Dict:
        """
        Compare plusieurs modèles et identifie le meilleur
        
        Args:
            metrics_list: Liste des dictionnaires de métriques
            
        Returns:
            Dictionnaire avec comparaison et meilleur modèle
        """
        if not metrics_list:
            return {}
        
        print("\n COMPARAISON DES MODÈLES")
        print("="*80)
        
        # En-tête du tableau
        headers = ["Modèle", "Hamming Loss", "Subset Acc", "Micro F1", "Macro F1"]
        print(f"{'Modèle':<25} | {'Hamming Loss':<12} | {'Subset Acc':<12} | {'Micro F1':<10} | {'Macro F1':<10}")
        print("-" * 80)
        
        # Afficher chaque modèle
        for metrics in metrics_list:
            print(f"{metrics.get('model', 'Unknown'):<25} | "
                  f"{metrics.get('hamming_loss', 0):<12.4f} | "
                  f"{metrics.get('subset_accuracy', 0):<12.4f} | "
                  f"{metrics.get('micro_f1', 0):<10.4f} | "
                  f"{metrics.get('macro_f1', 0):<10.4f}")
        
        print("=" * 80)
        
        # Identifier le meilleur modèle (par Micro F1)
        best_metrics = max(metrics_list, key=lambda x: x.get('micro_f1', 0))
        print(f"\n MEILLEUR MODÈLE: {best_metrics.get('model', 'Unknown')} "
              f"(Micro F1: {best_metrics.get('micro_f1', 0):.4f})")
        
        # Retourner les résultats
        return {
            'comparison_table': metrics_list,
            'best_model': best_metrics.get('model', 'Unknown'),
            'best_micro_f1': best_metrics.get('micro_f1', 0),
            'best_metrics': best_metrics
        }
    
    def _print_metrics_report(self, metrics: Dict):
        """Affiche un rapport détaillé des métriques"""
        print(f"\n{'='*50}")
        print(f"RÉSULTATS - {metrics.get('model', 'Modèle')}")
        print(f"{'='*50}")
        print(f"Hamming Loss:              {metrics.get('hamming_loss', 0):.4f}")
        print(f"Subset Accuracy:           {metrics.get('subset_accuracy', 0):.4f}")
        print(f"--- Métriques par exemple ---")
        print(f"Precision:                 {metrics.get('example_precision', 0):.4f}")
        print(f"Recall:                    {metrics.get('example_recall', 0):.4f}")
        print(f"F1-Score:                  {metrics.get('example_f1', 0):.4f}")
        print(f"--- Métriques par label ---")
        print(f"Macro Precision:           {metrics.get('macro_precision', 0):.4f}")
        print(f"Macro Recall:              {metrics.get('macro_recall', 0):.4f}")
        print(f"Macro F1:                  {metrics.get('macro_f1', 0):.4f}")
        print(f"Micro Precision:           {metrics.get('micro_precision', 0):.4f}")
        print(f"Micro Recall:              {metrics.get('micro_recall', 0):.4f}")
        print(f"Micro F1:                  {metrics.get('micro_f1', 0):.4f}")
        print(f"--- Statistiques ---")
        print(f"Moyenne labels réels:      {metrics.get('avg_true_labels', 0):.2f}")
        print(f"Moyenne labels prédits:    {metrics.get('avg_pred_labels', 0):.2f}")
        print(f"{'='*50}")