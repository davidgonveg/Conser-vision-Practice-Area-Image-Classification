#!/usr/bin/env python3
"""
Script para descubrir qué archivos Python reales tienes disponibles
en tu proyecto tai-park-classifier.
"""

import os
import sys
from pathlib import Path

def find_python_files(directory="."):
    """Encuentra todos los archivos Python en el proyecto."""
    
    python_files = []
    notebooks = []
    configs = []
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'env', 'tai-park-env', 
                    '.ipynb_checkpoints', 'node_modules', '.idea', '.vscode'}
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, directory)
            
            if file.endswith('.py'):
                python_files.append(relative_path)
            elif file.endswith('.ipynb'):
                notebooks.append(relative_path)
            elif file.endswith(('.yaml', '.yml', '.json')):
                configs.append(relative_path)
    
    return python_files, notebooks, configs

def categorize_files(python_files):
    """Categoriza los archivos Python por tipo."""
    
    categories = {
        'scripts': [],
        'src': [],
        'tests': [],
        'notebooks': [],
        'root': [],
        'other': []
    }
    
    for file in python_files:
        if file.startswith('scripts/'):
            categories['scripts'].append(file)
        elif file.startswith('src/'):
            categories['src'].append(file)
        elif file.startswith('test'):
            categories['tests'].append(file)
        elif 'notebook' in file.lower() or file.endswith('.ipynb'):
            categories['notebooks'].append(file)
        elif '/' not in file:  # Root directory
            categories['root'].append(file)
        else:
            categories['other'].append(file)
    
    return categories

def analyze_executable_files(python_files):
    """Analiza qué archivos parecen ejecutables (scripts principales)."""
    
    executable_candidates = []
    
    for file in python_files:
        # Check if it's in scripts directory
        if file.startswith('scripts/'):
            executable_candidates.append(f"{file} (in scripts directory)")
        
        # Check for main patterns in filename
        if any(keyword in file.lower() for keyword in ['main', 'run', 'train', 'test', 'debug']):
            executable_candidates.append(f"{file} (contains executable keyword)")
        
        # Check if it's in root directory (often executable)
        if '/' not in file and file != '__init__.py':
            executable_candidates.append(f"{file} (in root directory)")
    
    return executable_candidates

def main():
    """Función principal."""
    
    print("🔍 DESCUBRIMIENTO DE ARCHIVOS REALES EN TAI-PARK-CLASSIFIER")
    print("=" * 65)
    
    # Find all files
    python_files, notebooks, configs = find_python_files()
    
    print(f"\n📊 RESUMEN:")
    print(f"   📄 Archivos Python: {len(python_files)}")
    print(f"   📓 Notebooks: {len(notebooks)}")
    print(f"   ⚙️  Config files: {len(configs)}")
    
    # Categorize Python files
    categories = categorize_files(python_files)
    
    print(f"\n🗂️  ARCHIVOS PYTHON POR CATEGORÍA:")
    for category, files in categories.items():
        if files:
            print(f"\n📁 {category.upper()}:")
            for file in sorted(files):
                print(f"   • {file}")
    
    # Show notebooks
    if notebooks:
        print(f"\n📓 NOTEBOOKS DISPONIBLES:")
        for notebook in sorted(notebooks):
            print(f"   • {notebook}")
    
    # Show configs
    if configs:
        print(f"\n⚙️  ARCHIVOS DE CONFIGURACIÓN:")
        for config in sorted(configs):
            print(f"   • {config}")
    
    # Analyze executable files
    executable_candidates = analyze_executable_files(python_files)
    
    print(f"\n🚀 POSIBLES ARCHIVOS EJECUTABLES:")
    if executable_candidates:
        for candidate in executable_candidates:
            print(f"   • {candidate}")
    else:
        print("   ❌ No se encontraron archivos ejecutables obvios")
    
    # Specific checks for expected files
    print(f"\n🎯 VERIFICACIÓN DE ARCHIVOS CLAVE:")
    
    key_files = [
        'scripts/train_model.py',
        'scripts/train_notebook_style.py', 
        'scripts/example_complete_pipeline.py',
        'scripts/debug_auto_trainer.py',
        'src/data/__init__.py',
        'src/models/model.py',
        'configs/base_config.yaml'
    ]
    
    for key_file in key_files:
        exists = key_file in python_files or os.path.exists(key_file)
        status = "✅ EXISTE" if exists else "❌ NO EXISTE"
        print(f"   {status}: {key_file}")
    
    # Recommendations
    print(f"\n💡 RECOMENDACIONES:")
    
    if 'scripts/debug_auto_trainer.py' in python_files:
        print("   ✅ debug_auto_trainer.py existe - puedes ejecutarlo para diagnóstico")
    
    if any('train' in f for f in python_files):
        training_files = [f for f in python_files if 'train' in f]
        print(f"   🎯 Archivos de entrenamiento encontrados: {', '.join(training_files)}")
    
    if notebooks:
        print(f"   📓 Tienes {len(notebooks)} notebooks - pueden ser la forma principal de ejecutar el código")
    
    if not any(f.startswith('scripts/') for f in python_files):
        print("   ⚠️  No hay directorio scripts/ - el proyecto puede usar notebooks principalmente")
    
    # Docker check
    docker_files = [f for f in os.listdir('.') if 'docker' in f.lower() or f == 'Dockerfile']
    if docker_files:
        print(f"   🐳 Archivos Docker encontrados: {', '.join(docker_files)}")
        print("      Los comandos python se ejecutarían dentro del contenedor Docker")
    
    print(f"\n🎯 CONCLUSIÓN:")
    if python_files:
        print(f"   ✅ Proyecto Python funcional con {len(python_files)} archivos")
        if executable_candidates:
            print("   🚀 Hay archivos ejecutables disponibles")
        else:
            print("   📓 Proyecto parece usar principalmente notebooks")
    else:
        print("   ❌ No se encontraron archivos Python")

if __name__ == '__main__':
    main()