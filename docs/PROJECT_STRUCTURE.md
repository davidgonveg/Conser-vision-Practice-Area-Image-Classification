# 🦁 Taï Park Species Classification - Estructura Completa del Proyecto

## 📁 **ESTRUCTURA DE ARCHIVOS - UBICACIONES EXACTAS**

```
tai-park-classifier/
├── 📄 PROJECT_STRUCTURE.md         # ✅ Esta guía
├── 📄 README.md                    # ✅ Ya existe
├── 📄 requirements.txt             # ✅ Ya existe
├── 📄 environment.yml              # ✅ Ya existe (opcional)
├── 📄 setup.py                     # ✅ Ya existe (opcional)
│
├── 📁 configs/                     # ⚙️ Configuraciones
│   ├── base_config.yaml           # ✅ Ya existe - configuración base
│   ├── notebook_replica.yaml      # ✅ NUEVO - replica exacta del notebook
│   └── competition_config.yaml    # ✅ NUEVO - optimizada para competición
│
├── 📁 scripts/                     # 🚀 Scripts ejecutables
│   ├── train_model.py              # ✅ NUEVO - script principal avanzado
│   ├── train_notebook_style.py    # ✅ Ya existe - estilo notebook simple
│   ├── evaluate_model.py          # ✅ NUEVO - evaluación completa
│   └── generate_submission.py     # ✅ NUEVO - generación de submissions
│
├── 📁 src/                         # 📚 Código fuente principal
│   ├── 📁 data/                    
│   │   ├── __init__.py            # ✅ Ya existe
│   │   ├── dataset.py             # ✅ Ya existe
│   │   ├── data_loader.py         # ✅ Ya existe
│   │   ├── transforms.py          # ✅ Ya existe
│   │   └── preprocessing.py       # ✅ Ya existe
│   │
│   ├── 📁 models/                  
│   │   ├── __init__.py            # ✅ Ya existe
│   │   └── model.py               # ✅ Ya existe
│   │
│   ├── 📁 training/               
│   │   ├── __init__.py            # ✅ ACTUALIZADO - configuraciones avanzadas
│   │   ├── trainer.py             # ✅ Ya existe
│   │   └── losses.py              # ✅ NUEVO - funciones de pérdida avanzadas
│   │
│   ├── 📁 evaluation/              
│   │   ├── __init__.py            # ✅ Ya existe (probablemente)
│   │   ├── evaluator.py           # ✅ Ya existe
│   │   └── metrics.py             # ✅ Ya existe
│   │
│   ├── 📁 inference/              
│   │   ├── __init__.py            # ✅ Ya existe (probablemente)
│   │   └── predictor.py           # ✅ Ya existe
│   │
│   └── 📁 utils/                  
│       ├── __init__.py            # ✅ Ya existe
│       ├── config.py              # ✅ NUEVO - gestión de configuración
│       ├── logging_utils.py       # ✅ Ya existe
│       └── helpers.py             # ✅ Ya existe
│
├── 📁 data/                        # 💾 Datos del proyecto
│   ├── raw/                       # Datos originales
│   ├── processed/                 # Datos procesados
│   └── submissions/               # Archivos de submission
│
├── 📁 results/                     # 📊 Resultados de experimentos
│   ├── models/                    # Modelos entrenados
│   ├── logs/                      # Logs de entrenamiento
│   ├── plots/                     # Visualizaciones
│   └── evaluation/                # Resultados de evaluación
│
└── 📁 docs/                        # 📖 Documentación
    ├── TRAIN_MODEL_USAGE.md       # ✅ NUEVO - guía de uso completa
    └── PROJECT_STRUCTURE.md       # ✅ NUEVO - esta guía
```

---

## 🆕 **ARCHIVOS NUEVOS CREADOS**

### **1. Scripts Principales** 
```bash
# ✅ CREAR estos archivos en scripts/
scripts/train_model.py              # Script de entrenamiento avanzado
scripts/evaluate_model.py           # Script de evaluación completa  
scripts/generate_submission.py      # Script para generar submissions
```

### **2. Configuraciones**
```bash
# ✅ CREAR estos archivos en configs/
configs/notebook_replica.yaml       # Replica exacta del notebook
configs/competition_config.yaml     # Configuración para competición
```

### **3. Módulos Core**
```bash
# ✅ CREAR estos archivos en src/
src/training/losses.py              # Funciones de pérdida avanzadas
src/utils/config.py                 # Gestión de configuración

# ✅ ACTUALIZAR este archivo
src/training/__init__.py            # Configuraciones de entrenamiento
```

### **4. Documentación**
```bash
# ✅ CREAR estos archivos en docs/
docs/TRAIN_MODEL_USAGE.md          # Guía completa de uso
docs/PROJECT_STRUCTURE.md          # Esta guía de estructura
```

---

## 🚀 **COMANDOS PARA CREAR LA ESTRUCTURA**

### **Paso 1: Crear Directorios**
```bash
# Crear directorios si no existen
mkdir -p configs
mkdir -p docs
mkdir -p results/{models,logs,plots,evaluation}
mkdir -p data/submissions
```

### **Paso 2: Crear Archivos Principales**
Los archivos ya están creados en los artifacts. Copiar el contenido a:

1. **`scripts/train_model.py`** ← Copiar contenido del artifact `train_model_script`
2. **`scripts/evaluate_model.py`** ← Copiar contenido del artifact `evaluate_script`  
3. **`scripts/generate_submission.py`** ← Copiar contenido del artifact `generate_submission`
4. **`src/training/losses.py`** ← Copiar contenido del artifact `focal_loss`
5. **`src/utils/config.py`** ← Copiar contenido del artifact `config_utility`
6. **`src/training/__init__.py`** ← Copiar contenido del artifact `training_init`
7. **`configs/notebook_replica.yaml`** ← Copiar contenido del artifact `notebook_config`
8. **`configs/competition_config.yaml`** ← Copiar contenido del artifact `competition_config`
9. **`docs/TRAIN_MODEL_USAGE.md`** ← Copiar contenido del artifact `usage_examples`
10. **`docs/PROJECT_STRUCTURE.md`** ← Copiar contenido del artifact `complete_structure`

### **Paso 3: Hacer Scripts Ejecutables**
```bash
chmod +x scripts/train_model.py
chmod +x scripts/evaluate_model.py  
chmod +x scripts/generate_submission.py
```

---

## 🎯 **COMANDOS DE USO PRINCIPALES**

### **1. Entrenamiento Básico**
```bash
# Entrenamiento simple
python scripts/train_model.py

# Replica exacta del notebook (sin data leakage)
python scripts/train_model.py \
    --model resnet152 \
    --optimizer sgd \
    --learning-rate 0.01 \
    --momentum 0.909431 \
    --weight-decay 0.005 \
    --scheduler plateau \
    --freeze-backbone \
    --unfreeze-layers layer4 fc
```

### **2. Entrenamiento para Competición**
```bash
# Usar configuración de competición
python scripts/train_model.py \
    --config configs/competition_config.yaml \
    --experiment-name "competition_final"

# Entrenamiento avanzado personalizado
python scripts/train_model.py \
    --model efficientnet_b4 \
    --loss focal \
    --class-weights \
    --aggressive-aug \
    --mixed-precision \
    --sampler site_aware \
    --wandb
```

### **3. Evaluación**
```bash
# Evaluación básica
python scripts/evaluate_model.py \
    --model results/models/best_model.pth

# Evaluación completa con visualizaciones
python scripts/evaluate_model.py \
    --model results/models/best_model.pth \
    --save-plots \
    --detailed-analysis \
    --use-tta
```

### **4. Generar Submissions**
```bash
# Submission simple
python scripts/generate_submission.py \
    --model results/models/best_model.pth

# Submission con TTA y ensemble
python scripts/generate_submission.py \
    --ensemble results/models/model1.pth results/models/model2.pth \
    --use-tta \
    --output submissions/final_submission.csv
```

---

## 🔧 **VERIFICACIÓN DE LA INSTALACIÓN**

### **Test Rápido del Sistema Completo**
```bash
# 1. Verificar que todos los imports funcionan
python -c "from src.training.losses import FocalLoss; print('✅ Losses OK')"
python -c "from src.utils.config import Config; print('✅ Config OK')"

# 2. Test de entrenamiento rápido
python scripts/train_model.py --quick-test --dry-run

# 3. Test de configuración
python scripts/train_model.py --config configs/notebook_replica.yaml --dry-run

# 4. Verificar estructura de archivos
python -c "
import os
files = [
    'scripts/train_model.py',
    'scripts/evaluate_model.py', 
    'scripts/generate_submission.py',
    'src/training/losses.py',
    'src/utils/config.py',
    'configs/notebook_replica.yaml'
]
for f in files:
    status = '✅' if os.path.exists(f) else '❌'
    print(f'{status} {f}')
"
```

---

## 🏆 **FLUJO DE TRABAJO COMPLETO**

### **Para Experimentación Rápida**
```bash
# 1. Test rápido
python scripts/train_model.py --quick-test --fraction 0.1 --epochs 2

# 2. Desarrollo con datos reducidos  
python scripts/train_model.py --fraction 0.2 --epochs 10 --experiment-name "dev"

# 3. Evaluación del modelo de desarrollo
python scripts/evaluate_model.py --model results/models/dev/best_model.pth
```

### **Para Competición Seria**
```bash
# 1. Entrenamiento con configuración optimizada
python scripts/train_model.py \
    --config configs/competition_config.yaml \
    --wandb \
    --experiment-name "competition_v1"

# 2. Evaluación detallada
python scripts/evaluate_model.py \
    --model results/models/competition_v1/best_model.pth \
    --save-plots \
    --detailed-analysis

# 3. Generar submission final
python scripts/generate_submission.py \
    --model results/models/competition_v1/best_model.pth \
    --use-tta \
    --output submissions/competition_v1_tta.csv
```

### **Para Ensemble Avanzado**
```bash
# 1. Entrenar múltiples modelos
python scripts/train_model.py --model efficientnet_b3 --experiment-name "model_1"
python scripts/train_model.py --model efficientnet_b4 --experiment-name "model_2"  
python scripts/train_model.py --model resnet152 --experiment-name "model_3"

# 2. Generar ensemble submission
python scripts/generate_submission.py \
    --ensemble \
        results/models/model_1/best_model.pth \
        results/models/model_2/best_model.pth \
        results/models/model_3/best_model.pth \
    --ensemble-weights 0.4 0.4 0.2 \
    --use-tta \
    --output submissions/ensemble_final.csv
```

---

## 📊 **COMPARACIÓN: NOTEBOOK vs SISTEMA MODULAR**

| **Aspecto** | **Notebook Original** | **Sistema Modular** | **Mejora** |
|-------------|----------------------|---------------------|------------|
| **Data Split** | `train_test_split()` por imágenes ❌ | Split por sitios completos ✅ | **CRÍTICA** - Elimina data leakage |
| **Configuración** | Parámetros hardcodeados | YAML + args configurables | **MAYOR** - Flexibilidad total |
| **Modelos** | Solo ResNet152 | 15+ arquitecturas | **MAYOR** - Máxima versatilidad |
| **Pérdidas** | Solo CrossEntropy | 5+ funciones de pérdida | **MAYOR** - Para datos desbalanceados |
| **Augmentation** | Funciones fijas | Sistema modular + agresivo | **MAYOR** - Mejor generalización |
| **Monitoreo** | Print statements | TensorBoard + W&B + logs | **MAYOR** - Profesional |
| **Reproducibilidad** | Parcial | Seeds + determinístico | **MAYOR** - 100% reproducible |
| **Escalabilidad** | Script único | Sistema modular | **MAYOR** - Fácil extensión |
| **Evaluación** | Accuracy básica | 20+ métricas + plots | **MAYOR** - Análisis completo |
| **Submissions** | Manual | TTA + ensemble automático | **MAYOR** - Competición ready |

---

## 🎯 **CONFIGURACIONES PREDEFINIDAS**

### **1. Replica Exacta del Notebook** 
```bash
# Usa: configs/notebook_replica.yaml
python scripts/train_model.py --config configs/notebook_replica.yaml
```
**Características:**
- ResNet152 + SGD + momentum 0.909431
- Learning rate 0.01, weight decay 0.005
- ReduceLROnPlateau (patience=2, factor=0.72)
- Mismos parámetros exactos del notebook
- **PERO SIN data leakage** (split por sitios)

### **2. Competición Optimizada**
```bash
# Usa: configs/competition_config.yaml  
python scripts/train_model.py --config configs/competition_config.yaml
```
**Características:**
- EfficientNet-B4 + AdamW
- Focal Loss + class weights
- Aggressive augmentation
- Mixed precision + TTA
- Site-aware sampling

### **3. Configuraciones Rápidas por CLI**
```bash
# Desarrollo rápido
python scripts/train_model.py --model efficientnet_b0 --quick-test --fraction 0.1

# Científica reproducible  
python scripts/train_model.py --model resnet101 --optimizer sgd --deterministic

# Especies raras
python scripts/train_model.py --loss focal --focal-gamma 3.0 --class-weights --sampler balanced_batch
```

---

## 🔄 **MIGRACIÓN DESDE NOTEBOOK**

### **Si tienes un notebook funcionando:**

1. **Identifica parámetros del notebook:**
   ```python
   # Del notebook original
   model = resnet152(pretrained=True)
   optimizer = SGD(lr=0.01, momentum=0.909431, weight_decay=0.005)
   scheduler = ReduceLROnPlateau(patience=2, factor=0.72)
   ```

2. **Convierte a comando del sistema:**
   ```bash
   python scripts/train_model.py \
       --model resnet152 \
       --optimizer sgd \
       --learning-rate 0.01 \
       --momentum 0.909431 \
       --weight-decay 0.005 \
       --scheduler plateau \
       --scheduler-patience 2 \
       --scheduler-factor 0.72
   ```

3. **O crea config YAML personalizada:**
   ```yaml
   # mi_notebook_config.yaml
   model:
     name: "resnet152"
   training:
     optimizer: "sgd"
     learning_rate: 0.01
     momentum: 0.909431
     weight_decay: 0.005
   scheduler:
     type: "plateau"
     patience: 2  
     factor: 0.72
   ```

---

## 🚨 **TROUBLESHOOTING COMÚN**

### **Error: Módulos no encontrados**
```bash
# Solución 1: Instalar en modo desarrollo
pip install -e .

# Solución 2: Agregar al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solución 3: Verificar que estás en el directorio raíz
pwd  # Debe mostrar el directorio tai-park-classifier/
```

### **Error: CUDA out of memory**  
```bash
# Reducir batch size y usar mixed precision
python scripts/train_model.py \
    --batch-size 8 \
    --mixed-precision \
    --num-workers 2
```

### **Error: Archivos de configuración**
```bash
# Verificar que el archivo existe
ls configs/notebook_replica.yaml

# Usar ruta absoluta si es necesario
python scripts/train_model.py --config $(pwd)/configs/notebook_replica.yaml
```

### **Error: Datos no encontrados**
```bash
# Verificar estructura de datos
ls data/raw/
# Debe tener: train_features/, test_features/, *.csv

# Especificar ruta explícitamente
python scripts/train_model.py --data-dir /path/to/your/data
```

---

## 📈 **MONITOREO Y ANÁLISIS**

### **Durante el Entrenamiento**
```bash
# Ver logs en tiempo real
tail -f results/logs/[experiment_name]/training.log

# TensorBoard
tensorboard --logdir results/logs/[experiment_name]/tensorboard

# Weights & Biases (si está configurado)
# Ir a https://wandb.ai/tu_proyecto
```

### **Después del Entrenamiento**
```bash
# Evaluación completa
python scripts/evaluate_model.py \
    --model results/models/[experiment]/best_model.pth \
    --save-plots \
    --detailed-analysis

# Ver métricas guardadas
cat results/models/[experiment]/training_history.json
```

---

## 🎉 **BENEFICIOS DEL SISTEMA COMPLETO**

### **✅ Para Competiciones**
- **TTA automático** para mejor precisión
- **Ensemble de modelos** fácil
- **Validación de submissions** automática
- **Métricas de competición** (log loss, etc.)

### **✅ Para Investigación**
- **Reproducibilidad total** con seeds
- **Experimentos trazables** con W&B
- **Comparación de modelos** automática
- **Análisis detallado** por clases

### **✅ Para Producción**
- **Código modular** fácil de mantener
- **Configuración externa** sin cambios de código
- **Logging profesional** para debugging
- **Checkpoints automáticos** para recuperación

### **✅ Para Aprendizaje**
- **Múltiples architecturas** para experimentar
- **Pérdidas avanzadas** para casos especiales
- **Visualizaciones automáticas** para entender datos
- **Documentación completa** para referencia

---

## 🎯 **PRÓXIMOS PASOS RECOMENDADOS**

### **1. Setup Inicial (5 min)**
```bash
# Crear todos los archivos del sistema
# Verificar que imports funcionan
python -c "from src.training.losses import FocalLoss; print('OK')"
```

### **2. Test Rápido (2 min)**  
```bash
# Verificar que todo funciona
python scripts/train_model.py --quick-test --dry-run
```

### **3. Primer Experimento Real (30 min)**
```bash
# Entrenamiento con datos reducidos
python scripts/train_model.py \
    --fraction 0.2 \
    --epochs 10 \
    --experiment-name "first_test"
```

### **4. Competición Seria (2-4 horas)**
```bash
# Entrenamiento completo optimizado
python scripts/train_model.py \
    --config configs/competition_config.yaml \
    --experiment-name "competition_final"
```

---

## 📚 **RECURSOS ADICIONALES**

- **Documentación detallada**: `docs/TRAIN_MODEL_USAGE.md`
- **Ejemplos de configuración**: `configs/`  
- **Logs de ejemplo**: `results/logs/`
- **Código fuente comentado**: `src/`

---

## 🎊 **¡SISTEMA COMPLETO LISTO!**

Ahora tienes un sistema de clasificación de especies **de nivel profesional** que:

🎯 **Replica exactamente tu notebook exitoso** pero sin data leakage  
🚀 **Soporta 15+ arquitecturas de modelos** para experimentación  
⚙️ **Es completamente configurable** via YAML o argumentos  
📊 **Incluye monitoreo profesional** con TensorBoard y W&B  
🏆 **Está optimizado para competiciones** con TTA y ensembles  
🔬 **Es perfecto para investigación** con reproducibilidad total  
📈 **Escala para producción** con código modular robusto

**¡A entrenar modelos de clase mundial!** 🦁🚀