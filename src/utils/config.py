import os

# Paramètres du modèle


# # Features
# MAX_FEATURES = 5000  # TF-IDF
# MIN_DF = 5
# MAX_DF = 0.8
# NGRAM_RANGE = (1, 2)

# # Top catégories
# TOP_CATEGORIES = 10



import yaml
import os
from pathlib import Path

class Config:
    """Gestion centralisée de la configuration du projet"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path
        self.config = self._load_config()
    
    def _load_config(self):
        """Charger la configuration depuis YAML"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._default_config()
    
    def _default_config(self):
        """Configuration par défaut"""
        return {
            'project': {
                'name': 'arxivArticle-multilabel-classification',
                'version': '1.0.0'
            },
            'spark': {
                'app_name': 'ArXivArticle-Classification',
                'master': 'local[*]',
                'memory': '4g',
                'executor_memory': '2g'
            },
            'data': {
                'raw_path': 'data/raw',
                'processed_path': 'data/processed',
                'models_path': 'data/models'
            },
            'preprocessing': {
                'min_df': 5,
                'max_df': 0.8,
                'vocab_size': 10000,
                'test_size': 0.2,
                'val_size': 0.1
            },
            'modeling': {
                'max_doccuments': 30000 , # Réduire si problèmes mémoire 20000
                'random_state': 42,
                'num_partitions': 4
            }
        }