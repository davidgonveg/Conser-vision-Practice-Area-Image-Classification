#!/usr/bin/env python3
"""
Script de diagnóstico para problemas del auto-entrenamiento
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_models_one_epoch():
    """Prueba cada modelo con 1 época para identificar cuáles fallan."""
    
    print("\n🎯 Probando modelos individuales con 1 época...")
    
    # Modelos a probar (los que estaban fallando)
    models_to_test = [
        'resnet50',
        'resnet101', 
        'efficientnet_b0',
        'efficientnet_b1',
        'efficientnet_b2',
        'efficientnet_b3',
        'efficientnet_b4',
        'convnext_tiny'
    ]
    
    results = {}
    
    for model in models_to_test:
        print(f"\n🔍 Probando {model}...")
        
        # Comando básico de 1 época
        cmd = [
            sys.executable, 'scripts/train_model.py',
            '--model', model,
            '--epochs', '1',
            '--batch-size', '8',  # Batch size pequeño para evitar OOM
            '--learning-rate', '0.001',
            '--experiment-name', f'debug_{model}',
            '--quick-test'  # Sin dry-run, que entrene de verdad
        ]
        
        print(f"   Comando: {' '.join(cmd[-6:])}")  # Solo los argumentos relevantes
        
        try:
            print(f"   🚀 Ejecutando: {' '.join(cmd[-6:])}")  # Solo los argumentos relevantes
            
            # Ejecutar CON logs visibles en tiempo real
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=False,  # ¡AQUÍ! Mostrar logs en tiempo real
                text=True,
                timeout=300  # 5 minutos máximo por modelo
            )
            
            print(f"\n   📊 Resultado del proceso: Return code = {result.returncode}")
            
            if result.returncode == 0:
                print(f"   ✅ {model} FUNCIONA")
                results[model] = 'OK'
            else:
                print(f"   ❌ {model} FALLÓ (return code: {result.returncode})")
                results[model] = 'FAIL'
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {model} TIMEOUT")
            results[model] = 'TIMEOUT'
        
        except Exception as e:
            print(f"   💥 {model} ERROR: {e}")
            results[model] = f'ERROR: {e}'
    
    return results

def test_specific_arguments():
    """Prueba argumentos específicos que pueden estar causando problemas."""
    
    print("\n🔧 Probando argumentos específicos...")
    
    # Argumentos a probar
    args_to_test = [
        # Básico
        [],
        # Focal loss
        ['--focal-loss'],
        # Aggressive aug
        ['--aggressive-aug'],
        # Class weights
        ['--class-weights'],
        # Mixed precision
        ['--mixed-precision'],
        # Combination
        ['--focal-loss', '--class-weights'],
        # Samplers (los que estaban fallando)
        # ['--sampler', 'site_aware'],
        # ['--sampler', 'weighted'],
        # ['--sampler', 'balanced_batch']
    ]
    
    results = {}
    
    for i, extra_args in enumerate(args_to_test):
        test_name = f"args_test_{i}_{('_'.join(extra_args)).replace('--', '')}"
        print(f"\n🔍 Probando argumentos: {' '.join(extra_args) if extra_args else '(básico)'}")
        
        cmd = [
            sys.executable, 'scripts/train_model.py',
            '--model', 'resnet50',  # Modelo que sabemos que funciona
            '--epochs', '1',
            '--batch-size', '8',
            '--learning-rate', '0.001',
            '--experiment-name', test_name,
            '--quick-test'
        ] + extra_args
        
        try:
            print(f"   🚀 Ejecutando argumentos: {' '.join(extra_args) if extra_args else '(básico)'}")
            
            # Ejecutar CON logs visibles en tiempo real
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=False,  # ¡AQUÍ! Mostrar logs en tiempo real
                text=True,
                timeout=180  # 3 minutos
            )
            
            print(f"\n   📊 Resultado del proceso: Return code = {result.returncode}")
            
            if result.returncode == 0:
                print(f"   ✅ Argumentos FUNCIONAN")
                results[test_name] = 'OK'
            else:
                print(f"   ❌ Argumentos FALLAN")
                results[test_name] = 'FAIL'
        
        except Exception as e:
            print(f"   💥 ERROR: {e}")
            results[test_name] = f'ERROR: {e}'
    
    return results

def test_imports():
    """Prueba las importaciones básicas."""
    
    print("\n🔍 Verificando importaciones...")
    
    try:
        from src.utils.config import Config
        print("✅ Config import OK")
        
        from src.data import DataLoaderManager
        print("✅ DataLoaderManager import OK")
        
        from src.models.model import create_model
        print("✅ create_model import OK")
        
        # Test config loading
        config = Config("configs/base_config.yaml")
        print("✅ Config loading OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_data_availability():
    """Verifica que los datos estén disponibles."""
    
    print("\n🔍 Verificando datos...")
    
    data_dir = project_root / "data" / "raw"
    
    required_files = [
        "train_features.csv",
        "train_labels.csv",
        "test_features.csv"
    ]
    
    all_good = True
    
    for file in required_files:
        file_path = data_dir / file
        if file_path.exists():
            print(f"✅ {file} existe")
        else:
            print(f"❌ {file} NO existe")
            all_good = False
    
    # Check validation sites
    val_sites = project_root / "data" / "processed" / "validation_sites.csv"
    if val_sites.exists():
        print(f"✅ validation_sites.csv existe")
    else:
        print(f"❌ validation_sites.csv NO existe")
        all_good = False
    
    return all_good

def main():
    """Función principal de diagnóstico."""
    
    print("🚀 Iniciando diagnóstico COMPLETO del auto-entrenamiento")
    print("🔍 NOTA: Verás TODOS los logs de entrenamiento en tiempo real")
    print("=" * 60)
    
    # 1. Test imports
    imports_ok = test_imports()
    
    # 2. Check data
    data_ok = check_data_availability()
    
    if not all([imports_ok, data_ok]):
        print("\n🚨 Problemas básicos encontrados. Arreglar antes de continuar.")
        return
    
    # 3. Test cada modelo individual
    print(f"\n{'='*60}")
    print("🎯 FASE 1: PROBANDO MODELOS INDIVIDUALES (1 época cada uno)")
    print("🕐 Esto puede tardar 15-30 minutos total...")
    print("="*60)
    model_results = test_models_one_epoch()
    
    # 4. Test argumentos específicos
    print(f"\n{'='*60}")
    print("🔧 FASE 2: PROBANDO ARGUMENTOS ESPECÍFICOS")
    print("🕐 Esto tardará ~10-15 minutos más...")
    print("="*60)
    args_results = test_specific_arguments()
    
    # 5. Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN COMPLETO DEL DIAGNÓSTICO")
    print("="*60)
    
    print(f"Importaciones: {'✅ OK' if imports_ok else '❌ FAIL'}")
    print(f"Datos: {'✅ OK' if data_ok else '❌ FAIL'}")
    
    print(f"\n🧠 RESULTADOS DE MODELOS:")
    working_models = []
    failing_models = []
    
    for model, status in model_results.items():
        status_icon = "✅" if status == "OK" else "❌"
        print(f"   {status_icon} {model}: {status}")
        
        if status == "OK":
            working_models.append(model)
        else:
            failing_models.append(model)
    
    print(f"\n🔧 RESULTADOS DE ARGUMENTOS:")
    working_args = []
    failing_args = []
    
    for test_name, status in args_results.items():
        status_icon = "✅" if status == "OK" else "❌"
        args_display = test_name.replace('args_test_', '').replace('_', ' ')
        print(f"   {status_icon} {args_display}: {status}")
        
        if status == "OK":
            working_args.append(test_name)
        else:
            failing_args.append(test_name)
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    
    if working_models:
        print(f"✅ Modelos que FUNCIONAN: {', '.join(working_models)}")
        print(f"   -> Usar estos modelos en el auto-entrenamiento")
    
    if failing_models:
        print(f"❌ Modelos que FALLAN: {', '.join(failing_models)}")
        print(f"   -> Evitar estos modelos o investigar errores específicos")
    
    if working_args:
        print(f"✅ Argumentos que FUNCIONAN: Usar combinaciones exitosas")
    
    if failing_args:
        print(f"❌ Argumentos problemáticos: Evitar o usar alternativas")
    
    print(f"\n🔄 PRÓXIMOS PASOS:")
    if working_models:
        print(f"1. Actualizar auto_train_multiple.py para usar solo modelos que funcionan")
        print(f"2. Usar argumentos que se probaron exitosamente")
        print(f"3. Ejecutar auto-entrenamiento con configuración segura")
    else:
        print(f"1. Investigar errores específicos de los modelos")
        print(f"2. Verificar instalación de timm y dependencias")
        print(f"3. Probar con entorno virtual limpio")
    
    return {
        'models': model_results,
        'args': args_results,
        'working_models': working_models,
        'failing_models': failing_models
    }

if __name__ == "__main__":
    main()