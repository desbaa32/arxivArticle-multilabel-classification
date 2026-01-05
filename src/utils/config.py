import yaml
import os
from pathlib import Path

class Config:
    """Gestion centralisée de la configuration du projet"""
    
    def __init__(self, config_path="config.yaml"):
        self.project_root = Path(__file__).parent.parent
        self.config_path = self.project_root / config_path
        self.config = self._load_config()
        self._setup_paths
    
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
                'master': 'spark://tawfekh-d:7077',
                'memory': '4g',
                'executor_memory': '2g'
            },
            'data': {
                'raw_path': 'data/raw/arxiv-metadata-oai-snapshot.json',
                'processed_path': 'data/processed/',
                'sample_size': 50000,
                'top_categories': 30
            }
            # ,
            # 'preprocessing': {
            #     'min_df': 5,
            #     'max_df': 0.8,
            #     'vocab_size': 10000,
            #     'test_size': 0.2,
            #     'val_size': 0.1
            # },
            # 'modeling': {
            #     'max_doccuments': 30000 , # Réduire si problèmes mémoire 20000
            #     'random_state': 42,
            #     'num_partitions': 4
            # }
        }
    def _setup_paths(self):
        """Crée les répertoires nécessaires"""
        paths = [
            'data/processed',
            'data/interim',
            'models',
            'logs',
            'reports/figures',
            'reports/tables'
        ]
        
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    def get_spark_config(self):
        """Retourne la configuration Spark"""
        return self.config.get('spark', {})
    
    def get_data_config(self):
        """Retourne la configuration des données"""
        return self.config.get('data', {})
    
    def get_model_config(self, model_type):
        """Retourne la configuration d'un modèle spécifique"""
        models_config = self.config.get('models', {})
        return models_config.get(model_type, {})
    
    def __getitem__(self, key):
        """Accès aux valeurs de configuration"""
        return self.config.get(key, {})