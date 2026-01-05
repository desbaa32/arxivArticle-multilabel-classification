# cleanup_git.sh
#!/bin/bash

echo " NETTOYAGE DES FICHIERS VOLUMINEUX DU PROJET"
echo "================================================"

# 1. Lister les gros fichiers
echo " Recherche des fichiers volumineux (>100MB)..."

# Trouver les fichiers > 100MB
find . -type f -size +100M ! -path "./.git/*" ! -path "./venv/*" ! -path "./.venv/*" | while read file; do
    size=$(du -h "$file" | cut -f1)
    echo " Gros fichier: $file ($size)"
done

# 2. Nettoyer les fichiers de données
echo ""
echo " Suppression des fichiers de données du cache Git..."

# Supprimer les données brutes et intermédiaires du cache
git rm --cached -r data/raw/ 2>/dev/null || true
git rm --cached -r data/processed/ 2>/dev/null || true
git rm --cached -r data/interim/ 2>/dev/null || true
git rm --cached -r models/ 2>/dev/null || true

# Supprimer les fichiers Parquet spécifiques
find . -name "*.parquet" -exec git rm --cached {} \; 2>/dev/null || true
find . -name "*.json" -size +10M -exec git rm --cached {} \; 2>/dev/null || true

echo "✅ Fichiers volumineux supprimés du cache"

# 3. Ajouter tout au .gitignore
echo ""
echo "Configuration du .gitignore..."
cat > .gitignore << 'EOF'
# DONNÉES VOLUMINEUSES - NE PAS COMMITER
data/raw/
data/processed/
data/interim/
*.parquet
*.json
*.csv
*.pkl
*.joblib
*.h5
*.model

# MODÈLES
models/

# LOGS ET TEMPORAIRES
logs/
temp/
tmp/
*.log

# ENVIRONNEMENTS
venv/
.env
.venv/

# CACHE ET BUILD
__pycache__/
*.pyc
.ipynb_checkpoints/
.spark/
EOF

echo " .gitignore mis à jour"
