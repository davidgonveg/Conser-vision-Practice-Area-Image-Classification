# 🦁 Tai Park Wildlife Classification - Usage Guide

Este sistema modular replica **exactamente** la lógica exitosa de tu notebook, pero organizada en módulos reutilizables y escalables.

## 🚀 Quick Start

### 1. Ejecutar Training Completo (Replica del Notebook)

```bash
# Entrenar exactamente como el notebook
python scripts/train_notebook_style.py

# Con parámetros personalizados
python scripts/train_notebook_style.py \
    --data_dir data/raw \
    --num_epochs 5 \
    --batch_size 64 \
    --fraction 1.0 \
    --random_state 1
```

### 2. Test Rápido para Desarrollo

```bash
# Test rápido con 10% de datos y 1 época
python scripts/train_notebook_style.py --quick_test
```

## 📋 Estructura del Sistema

```
tai-park-classifier/
├── src/
│   ├── data/
│   │   └── dataset.py          # ✅ Dataset con split por sitios
│   ├── models/
│   │   └── model.py            # ✅ ResNet152 + clasificación head
│   ├── training/
│   │   └── trainer.py          # ✅ Training loop completo
│   └── utils/
│       └── helpers.py          # ✅ Visualización y análisis
└── scripts/
    └── train_notebook_style.py # ✅ Script principal
```

## 🎯 Funcionalidades Implementadas

### ✅ Exactamente del Notebook:
- **Dataset Loading**: `train_features.csv`, `train_labels.csv`, `test_features.csv`
- **Preprocessing**: `custom_preprocessing()` con color, brillo, contraste
- **Augmentation**: `data_augmentation()` con rotación, flip, color jitter
- **Model**: ResNet152 con solo layer4 descongelado
- **Training Head**: Linear(2048→1024→256→8) con BatchNorm y Dropout
- **Optimizer**: SGD (lr=0.01, momentum=0.909431, weight_decay=0.005)
- **Scheduler**: ReduceLROnPlateau (patience=2, factor=0.72)
- **Early Stopping**: Custom logic con tolerance=5
- **Loss Tracking**: Evaluación en quarter-steps
- **Dataset Recreation**: Cada época recrea dataset con augmentation

### ✅ Mejoras Implementadas:
- **🚨 Split por Sitios**: Evita data leakage (el notebook usaba stratified split)
- **📁 Estructura Modular**: Código organizado y reutilizable
- **📊 Visualizaciones**: Distribución de clases, sitios, samples
- **🔍 Análisis Detallado**: Métricas completas, confusion matrix
- **💾 Checkpoints**: Guardado automático del mejor modelo
- **📈 Plots**: Loss curves automáticos como el notebook

## 🔧 Uso Programático

### Entrenamiento Personalizado

```python
from src.data.dataset import TaiParkDatasetNotebookStyle
from src.models.model import create_notebook_model
from src.training.trainer import create_notebook_trainer
from src.utils.helpers import notebook_style_summary

# 1. Crear dataset manager (replica notebook)
dataset_manager = TaiParkDatasetNotebookStyle(
    data_dir="data/raw",
    fraction=1.0,              # frac del notebook
    random_state=1,            # random_state del notebook
    use_preprocessing=True,    # custom_preprocessing
    use_augmentation=True,     # data_augmentation  
    num_augmentations=2        # create_combined_dataset
)

# 2. Crear modelo (replica notebook)
model = create_notebook_model()

# 3. Crear trainer (replica notebook)
trainer = create_notebook_trainer(model, dataset_manager)

# 4. Entrenar (replica notebook)
loss_history = trainer.train(
    num_epochs=5,
    batch_size=64
)

# 5. Mostrar resumen como notebook
notebook_style_summary(trainer, dataset_manager, model_info, loss_history)
```

### Crear Dataset para Inferencia

```python
from src.data.dataset import create_test_dataloader_notebook_style

# Test dataset para submissions
test_dataloader = create_test_dataloader_notebook_style(
    data_dir="data/raw",
    batch_size=64
)

# Generar submission
from src.utils.helpers import create_submission_file

submission = create_submission_file(
    model=model,
    test_dataloader=test_dataloader,
    class_names=dataset_manager.species_labels,
    device=trainer.device,
    output_path="data/submissions/my_submission.csv"
)
```

### Análisis y Visualización

```python
from src.utils.helpers import (
    visualize_samples, 
    plot_class_distribution,
    plot_site_distribution,
    evaluate_model
)

# Visualizar muestras como notebook
visualize_samples(dataset_manager, save_path="results/plots/samples.png")

# Distribución de clases
plot_class_distribution(dataset_manager, save_path="results/plots/classes.png")

# Verificar split por sitios (no overlap)
no_leakage = plot_site_distribution(dataset_manager, save_path="results/plots/sites.png")
print(f"Sin data leakage: {no_leakage}")

# Evaluación completa del modelo
eval_results = evaluate_model(
    model=trainer.model,
    dataloader=eval_dataloader,
    device=trainer.device,
    class_names=dataset_manager.species_labels
)

print(f"Accuracy: {eval_results['accuracy']:.4f}")
print(f"Log Loss: {eval_results['log_loss']:.4f}")
```

## 🎛️ Parámetros de Configuración

### Dataset Parameters

```python
TaiParkDatasetNotebookStyle(
    data_dir="data/raw",           # Carpeta con CSVs e imágenes
    fraction=1.0,                  # Fracción de datos (notebook: frac)
    random_state=1,                # Seed (notebook usa 1)
    validation_sites_file=None,    # CSV con sitios de validación
    test_size=0.25,                # Tamaño split validación
    use_preprocessing=True,        # custom_preprocessing()
    use_augmentation=True,         # data_augmentation()
    num_augmentations=2            # Notebooks usa 2
)
```

### Model Parameters

```python
create_notebook_model()           # Configuración exacta del notebook

# O personalizado:
WildlifeClassifier(
    model_name='resnet152',        # resnet152, resnet50, efficientnet_b3, etc.
    num_classes=8,                 # 8 especies + blank
    pretrained=True,               # Usar pesos ImageNet
    freeze_layers=True,            # Congelar capas
    unfreeze_layers=['layer4'],    # Solo layer4 entrenable (notebook)
    dropout_rates=(0.5, 0.3),      # Dropout en head (notebook)
    hidden_sizes=(1024, 256)       # Tamaños hidden layers (notebook)
)
```

### Training Parameters

```python
trainer.train(
    num_epochs=5,                  # Épocas máximas
    batch_size=64,                 # Batch size (notebook)
    save_best_model=True           # Cargar mejor modelo al final
)

# Optimizer (automático, igual al notebook):
# SGD(lr=0.01, momentum=0.909431, weight_decay=0.005)

# Scheduler (automático, igual al notebook):
# ReduceLROnPlateau(patience=2, factor=0.72)

# Early Stopping (automático, igual al notebook):
# min_delta=0.0001, tolerance=5
```

## 📊 Outputs Generados

### Modelos y Checkpoints
```
results/models/
├── notebook_style_model.pth      # Mejor modelo entrenado
└── checkpoint_epoch_X.pth        # Checkpoints por época
```

### Visualizaciones
```
results/plots/
├── loss_curves.png               # Training/eval loss (como notebook)
├── sample_images.png             # Muestras por especie
├── class_distribution.png        # Distribución de clases
├── site_distribution.png         # Distribución de sitios
└── confusion_matrix.png          # Matriz de confusión
```

### Submissions
```
data/submissions/
└── submission_YYYY-MM-DD.csv     # Archivo para competición
```

## 🔍 Diferencias vs Notebook Original

### ✅ Mejoras Implementadas:

1. **🚨 CRÍTICO - Split por Sitios**: 
   - ❌ Notebook: `train_test_split(stratify=y)` → Data leakage
   - ✅ Nuevo: Split por sitios completos → Sin data leakage

2. **📁 Código Modular**:
   - ❌ Notebook: Todo en celdas mezcladas
   - ✅ Nuevo: Módulos separados y reutilizables

3. **🔍 Análisis Mejorado**:
   - ✅ Verificación automática de data leakage
   - ✅ Métricas detalladas (accuracy, log loss, confusion matrix)
   - ✅ Análisis por sitios y clases

4. **💾 Gestión de Modelos**:
   - ✅ Guardado automático del mejor modelo
   - ✅ Checkpoints con configuración completa
   - ✅ Carga fácil para inferencia

### 🎯 Funcionalidades Preservadas:

- ✅ **Preprocessing exacto**: `custom_preprocessing()` idéntico
- ✅ **Augmentation exacto**: `data_augmentation()` idéntico  
- ✅ **Arquitectura exacta**: ResNet152 + head personalizado
- ✅ **Training loop exacto**: Early stopping, scheduler, evaluación en quarter-steps
- ✅ **Hiperparámetros exactos**: lr, momentum, weight_decay del notebook
- ✅ **Dataset recreation**: Cada época recrea dataset con augmentation
- ✅ **Loss tracking exacto**: Mismo formato que notebook

## 🚀 Comandos Útiles

### Training Completo
```bash
# Entrenamiento completo como notebook
python scripts/train_notebook_style.py \
    --num_epochs 5 \
    --batch_size 64 \
    --save_model_path results/models/my_model.pth

# Con validación por sitios específicos
python scripts/train_notebook_style.py \
    --validation_sites_file data/processed/validation_sites.csv
```

### Experimentación
```bash
# Experimento rápido (10% datos, 1 época)
python scripts/train_notebook_style.py \
    --fraction 0.1 \
    --num_epochs 1 \
    --batch_size 32

# Sin data augmentation (solo para testing)
python scripts/train_notebook_style.py \
    --num_augmentations 0
```

### Diferentes Modelos
```python
# ResNet50 en lugar de ResNet152
model = create_model(
    model_name='resnet50',
    num_classes=8,
    pretrained=True
)

# EfficientNet-B3  
model = create_model(
    model_name='efficientnet_b3',
    num_classes=8,
    pretrained=True
)
```

## 🎯 Próximos Pasos

1. **Ejecutar training completo**:
   ```bash
   python scripts/train_notebook_style.py
   ```

2. **Verificar resultados**:
   - Revisa `results/plots/loss_curves.png`
   - Verifica que no hay data leakage en site distribution
   - Compara loss final con tu notebook

3. **Generar submission**:
   ```python
   # Cargar modelo entrenado y crear submission
   submission = create_submission_file(...)
   ```

4. **Experimentar con variaciones**:
   - Diferentes arquitecturas (EfficientNet, etc.)
   - Diferentes hiperparámetros
   - Ensembles de modelos

## ⚡ Tips de Rendimiento

- **GPU Memory**: Si tienes problemas de memoria, reduce `batch_size`
- **Speed**: Para desarrollo rápido, usa `--fraction 0.1`
- **Reproducibilidad**: Siempre usa el mismo `random_state`
- **Monitoring**: Los logs te muestran progreso detallado

¡El sistema está listo para replicar y mejorar tus resultados del notebook! 🚀