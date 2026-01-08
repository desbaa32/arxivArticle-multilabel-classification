#  Classification Multi-Label arXiv avec PySpark

Système de classification automatique d'articles scientifiques utilisant PySpark et Machine Learning.

## 📦 Installation Rapide

```bash
# Cloner le projet
git clone <votre-repo>
cd arxiv-multilabel-classification



# Télécharger les données arXiv (Kaggle)
# Placer dans data/raw/
```

## 🚀 Utilisation

### Notebooks Jupyter (Recommandé)

```bash
jupyter notebook
```

Exécuter dans l'ordre :
1. `01_data_exploration.ipynb` - Explorer les données
2. `02_preprocessing.ipynb` - Nettoyer
3. `03_feature_engineering.ipynb` - Créer les features
4. `04_model_training.ipynb` - Entraîner
5. `05_evaluation.ipynb` - Évaluer

### Scripts Python

```bash
python scripts/run_loader.py
python scripts/run_preprocessing.py
python scripts/run_feature_engineering.py
python scripts/run_model_training.py
```

## 📊 Architecture

```
arxiv-multilabel-classification/
├── data/
│   ├── raw/           # Données brutes
│   ├── processed/     # Données nettoyées
│   ├── models/        # Modèles entraînés
│   └── results/       # Résultats
├── src/
│   ├── data/          # Chargement et prétraitement
│   ├── features/      # Feature engineering
│   ├── models/        # Binary Relevance & Classifier Chains
│   └── utils/         # Métriques et visualisations
├── notebooks/         # Notebooks Jupyter
└── scripts/           # Scripts d'exécution
```

## 🧠 Modèles

### 1. Binary Relevance
- Transforme le problème multi-label en N problèmes binaires
- Modèle : `LogisticRegression`
- Rapide et parallélisable

### 2. Classifier Chains
- Modélise les dépendances entre labels
- Modèle : `RandomForestClassifier`
- Performances supérieures


## 🛠️ Configuration

Éditer `config.yaml` :

```yaml
data:
  sample_size: 50000  # Nombre d'articles

features:
  vocab_size: 5000    # Taille vocabulaire TF-IDF

models:
  top_n_categories: 15  # Nombre de labels
```

## 📝 Licence

MIT License

## 👥 Auteurs

[GitHub](https://github.com/desbaa32)