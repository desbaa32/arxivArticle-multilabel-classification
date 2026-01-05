# scripts/create_init_files.py
import os
from pathlib import Path

def create_init_files(base_dir=Path(__file__).parent.parent):
    """Crée automatiquement tous les fichiers __init__.py"""
    
    for root, dirs, files in os.walk(base_dir):
        # Ignorer certains dossiers
        if any(x in root for x in ['.git', '__pycache__', '.ipynb_checkpoints']):
            continue
        
        # Vérifier si c'est un dossier Python (contient des .py)
        py_files = [f for f in files if f.endswith('.py')]
        
        if py_files:  # Dossier contient des fichiers Python
            init_file = os.path.join(root, '__init__.py')
            
            if not os.path.exists(init_file):
                # Déterminer le nom du package
                rel_path = os.path.relpath(root, base_dir)
                package_name = rel_path.replace(os.sep, '.') if rel_path != '.' else 'main'
                
                # Contenu du __init__.py
                content = f'''"""
Package: {package_name}
Generated automatically
"""

__all__ = []
'''
                
                with open(init_file, 'w') as f:
                    f.write(content)
                
                print(f" Créé: {init_file}")

if __name__ == "__main__":
    create_init_files()