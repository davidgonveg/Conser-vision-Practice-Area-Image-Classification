#!/usr/bin/env python3
"""
Diagnostic script to check the Taï Park project setup.
This script will identify what components are missing and provide specific guidance.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path):
    """Check if a file exists and return status."""
    path = Path(file_path)
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return exists, size

def check_import(module_path, class_or_function=None):
    """Check if a module and optionally a specific class/function can be imported."""
    try:
        if '.' in module_path:
            # Import from submodule
            module_parts = module_path.split('.')
            module = __import__(module_path, fromlist=[module_parts[-1]])
        else:
            module = __import__(module_path)
        
        if class_or_function:
            if hasattr(module, class_or_function):
                return True, f"✅ {module_path}.{class_or_function}"
            else:
                return False, f"❌ {module_path}.{class_or_function} not found in module"
        else:
            return True, f"✅ {module_path}"
    
    except ImportError as e:
        return False, f"❌ {module_path}: {str(e)}"
    except Exception as e:
        return False, f"❌ {module_path}: {str(e)}"

def check_file_contents(file_path, search_terms):
    """Check if a file contains specific terms (classes, functions)."""
    path = Path(file_path)
    if not path.exists():
        return {term: False for term in search_terms}
    
    try:
        content = path.read_text()
        results = {}
        for term in search_terms:
            # Look for class or function definitions
            patterns = [
                f"class {term}",
                f"def {term}",
                f"{term} =",
                f"from .* import.*{term}",
                f"import.*{term}"
            ]
            results[term] = any(pattern.lower() in content.lower() for pattern in patterns)
        return results
    except Exception:
        return {term: False for term in search_terms}

def main():
    print("🦁 Taï Park Species Classification - Complete Diagnosis")
    print("=" * 70)
    
    # Check Python path
    print(f"🐍 Python version: {sys.version}")
    print(f"📁 Current directory: {Path.cwd()}")
    
    print("\n" + "=" * 70)
    print("📂 CHECKING FILE STRUCTURE")
    print("=" * 70)
    
    # Check key files and their expected contents
    file_checks = {
        "src/__init__.py": [],
        "src/data/__init__.py": ["TaiParkDataset", "DataLoaderManager", "create_datasets"],
        "src/data/dataset.py": ["TaiParkDataset", "create_datasets", "create_test_dataset"],
        "src/data/data_loader.py": ["DataLoaderManager", "create_balanced_dataloader"],
        "src/data/transforms.py": ["get_train_transforms", "get_val_transforms"],
        "src/data/preprocessing.py": ["DatasetAnalyzer", "ImageValidator"],
        "src/models/__init__.py": ["create_model"],
        "src/models/model.py": ["create_model", "WildlifeClassifier"],
        "src/training/__init__.py": ["Trainer"],
        "src/training/trainer.py": ["Trainer"],
        "src/training/losses.py": ["FocalLoss"],
        "src/utils/__init__.py": [],
        "src/utils/config.py": ["Config"],
        "src/utils/logging_utils.py": ["setup_logging"],
        "scripts/train_model.py": ["main", "parse_args"],
        "scripts/train_notebook_style.py": ["main"],
        "configs/base_config.yaml": []
    }
    
    missing_files = []
    existing_files = []
    content_issues = {}
    
    for file_path, expected_content in file_checks.items():
        exists, size = check_file_exists(file_path)
        status = "✅" if exists else "❌"
        size_str = f"({size} bytes)" if exists else ""
        print(f"{status} {file_path} {size_str}")
        
        if exists:
            existing_files.append(file_path)
            if expected_content:
                # Check file contents
                content_results = check_file_contents(file_path, expected_content)
                missing_content = [term for term, found in content_results.items() if not found]
                if missing_content:
                    content_issues[file_path] = missing_content
                    print(f"   ⚠️  Missing: {', '.join(missing_content)}")
        else:
            missing_files.append(file_path)
    
    print("\n" + "=" * 70)
    print("🔗 CHECKING IMPORTS")
    print("=" * 70)
    
    # Check imports
    imports_to_check = [
        ("torch", None),
        ("torchvision", None),
        ("numpy", None),
        ("pandas", None),
        ("PIL", None),
        ("src.data", None),
        ("src.data.dataset", "TaiParkDataset"),
        ("src.data", "DataLoaderManager"),
        ("src.data", "create_datasets"),
        ("src.models", None),
        ("src.models.model", "create_model"),
        ("src.training", None),
        ("src.training.trainer", "Trainer"),
        ("src.training.losses", "FocalLoss"),
        ("src.utils.config", "Config"),
        ("src.utils.logging_utils", "setup_logging")
    ]
    
    import_issues = []
    working_imports = []
    for module_path, class_or_func in imports_to_check:
        success, message = check_import(module_path, class_or_func)
        print(message)
        
        if success:
            working_imports.append((module_path, class_or_func))
        else:
            import_issues.append((module_path, class_or_func, message))
    
    print("\n" + "=" * 70)
    print("📊 DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    print(f"📁 Files: {len(existing_files)}/{len(file_checks)} exist")
    print(f"🔗 Imports: {len(working_imports)}/{len(imports_to_check)} working")
    
    if not missing_files and not import_issues and not content_issues:
        print("\n🎉 SETUP IS PERFECT!")
        print("✅ All files exist, all imports work, all content is present")
        print("🚀 You can use: python scripts/train_model.py --quick-test --dry-run")
        return
    
    print("\n⚠️  ISSUES FOUND:")
    
    if missing_files:
        print(f"\n📁 Missing files ({len(missing_files)}):")
        for file in missing_files:
            print(f"   ❌ {file}")
    
    if content_issues:
        print(f"\n📄 Content issues ({len(content_issues)}):")
        for file, missing_items in content_issues.items():
            print(f"   ⚠️  {file}: missing {', '.join(missing_items)}")
    
    if import_issues:
        print(f"\n🔗 Import issues ({len(import_issues)}):")
        for module, class_func, message in import_issues:
            print(f"   {message}")
    
    print("\n" + "=" * 70)
    print("🔧 STEP-BY-STEP SOLUTION PLAN")
    print("=" * 70)
    
    # Prioritized solution steps
    solution_steps = []
    
    # Step 1: Core missing files
    core_missing = [f for f in missing_files if 'data_loader.py' in f or 'transforms.py' in f]
    if core_missing:
        solution_steps.append({
            'priority': 1,
            'title': 'Create core data files',
            'files': core_missing,
            'description': 'These are essential for DataLoaderManager and transforms'
        })
    
    # Step 2: Content issues in existing files
    if content_issues:
        solution_steps.append({
            'priority': 2,
            'title': 'Fix existing files',
            'files': list(content_issues.keys()),
            'description': 'Add missing classes/functions to existing files'
        })
    
    # Step 3: Other missing files
    other_missing = [f for f in missing_files if f not in core_missing]
    if other_missing:
        solution_steps.append({
            'priority': 3,
            'title': 'Create additional files',
            'files': other_missing,
            'description': 'These files add extra functionality'
        })
    
    for i, step in enumerate(solution_steps, 1):
        print(f"\n🔧 STEP {step['priority']}: {step['title']}")
        print(f"   📝 {step['description']}")
        print("   📁 Files to create/fix:")
        for file in step['files']:
            if file in missing_files:
                print(f"      ❌ CREATE: {file}")
            elif file in content_issues:
                missing_items = content_issues[file]
                print(f"      ⚠️  FIX: {file} (add: {', '.join(missing_items)})")
    
    print(f"\n💡 IMMEDIATE SOLUTION:")
    print("   Most critical: Create src/data/data_loader.py with DataLoaderManager class")
    print("   Second: Create src/data/transforms.py with get_train_transforms, get_val_transforms")
    print("   Third: Fix src/data/dataset.py to export TaiParkDataset properly")
    
    print(f"\n🚀 QUICK START:")
    print("   While fixing the above, you can use the working script:")
    print("   python scripts/train_notebook_style.py --help")


if __name__ == "__main__":
    main()