# 🦁 Train Model Script - Guía Completa de Uso

El script `train_model.py` es la herramienta más avanzada y configurable para entrenar modelos en el proyecto Taï Park Species Classification.

## 🚀 Comandos de Ejemplo

### 1. **Entrenamiento Básico**
```bash
# Entrenamiento simple con configuración por defecto
python scripts/train_model.py

# Con modelo y parámetros básicos
python scripts/train_model.py \
    --model efficientnet_b3 \
    --epochs 50 \
    --batch-size 32
```

### 2. **Replica Exacta del Notebook (ResNet152 + SGD)**
```bash
# Replica exacta del notebook solution.ipynb pero SIN data leakage
python scripts/train_model.py \
    --model resnet152 \
    --optimizer sgd \
    --learning-rate 0.01 \
    --momentum 0.909431 \
    --weight-decay 0.005 \
    --scheduler plateau \
    --scheduler-patience 2 \
    --scheduler-factor 0.72 \
    --freeze-backbone \
    --unfreeze-layers layer4 fc \
    --epochs 5 \
    --batch-size 64 \
    --experiment-name "notebook_replica"
```

### 3. **Entrenamiento Avanzado para Competición**
```bash
# Configuración óptima para competición
python scripts/train_model.py \
    --model efficientnet_b4 \
    --optimizer adamw \
    --learning-rate 0.001 \
    --weight-decay 0.01 \
    --scheduler cosine \
    --loss focal \
    --focal-gamma 2.0 \
    --class-weights \
    --aggressive-aug \
    --mixed-precision \
    --epochs 100 \
    --batch-size 64 \
    --sampler site_aware \
    --experiment-name "efficientnet_b4_competition"
```

### 4. **Entrenamiento para Especies Raras (Class Imbalance)**
```bash
# Optimizado para manejar desbalance de clases
python scripts/train_model.py \
    --model efficientnet_b5 \
    --loss weighted_focal \
    --focal-gamma 3.0 \
    --class-weights \
    --sampler balanced_batch \
    --aggressive-aug \
    --epochs 75 \
    --batch-size 32 \
    --experiment-name "rare_species_focus"
```

### 5. **Entrenamiento Rápido de Prueba**
```bash
# Test rápido con datos reducidos
python scripts/train_model.py \
    --quick-test \
    --fraction 0.1 \
    --epochs 2 \
    --batch-size 16 \
    --experiment-name "quick_test"

# Dry run para verificar configuración sin entrenar
python scripts/train_model.py \
    --dry-run \
    --model efficientnet_b3
```

### 6. **Entrenamiento con Monitoreo Completo**
```bash
# Con Weights & Biases y TensorBoard
python scripts/train_model.py \
    --model convnext_base \
    --wandb \
    --wandb-project "tai-park-advanced" \
    --wandb-tags experiment baseline convnext \
    --experiment-name "convnext_baseline" \
    --epochs 50 \
    --mixed-precision
```

### 7. **Diferentes Arquitecturas de Modelo**
```bash
# Vision Transformer
python scripts/train_model.py \
    --model vit_base_patch16_224 \
    --learning-rate 0.0005 \
    --optimizer adamw \
    --scheduler cosine

# ConvNeXt Large
python scripts/train_model.py \
    --model convnext_large \
    --batch-size 16 \
    --gradient-clip 1.0 \
    --mixed-precision

# EfficientNet B7 (modelo más grande)
python scripts/train_model.py \
    --model efficientnet_b7 \
    --batch-size 8 \
    --learning-rate 0.0005 \
    --mixed-precision \
    --epochs 30
```

### 8. **Configuraciones de Data Augmentation**
```bash
# Augmentation conservadora
python scripts/train_model.py \
    --horizontal-flip 0.3 \
    --rotation 10 \
    --brightness 0.1 \
    --contrast 0.1

# Augmentation agresiva para más datos
python scripts/train_model.py \
    --aggressive-aug \
    --horizontal-flip 0.7 \
    --rotation 20 \
    --brightness 0.3 \
    --contrast 0.3 \
    --color-jitter 0.4
```

## 📊 Parámetros Principales

### **Modelos Disponibles**
- `resnet50`, `resnet101`, `resnet152`
- `efficientnet_b0` a `efficientnet_b7`
- `convnext_base`, `convnext_large`
- `vit_base_patch16_224`

### **Optimizadores**
- `adam` (default): Funciona bien en general
- `adamw`: Mejor para modelos grandes (Transformers)
- `sgd`: Para replicar notebooks o entrenamiento clásico
- `rmsprop`: Alternativa robusta

### **Schedulers**
- `plateau`: Reduce LR cuando loss se estanca
- `cosine`: Cosine annealing, bueno para fine-tuning
- `step`: Reduce LR cada X épocas
- `exponential`: Decaimiento exponencial
- `none`: Sin scheduler

### **Funciones de Pérdida**
- `cross_entropy`: Estándar
- `focal`: Para desbalance de clases
- `label_smoothing`: Previene overfitting

### **Estrategias de Sampling**
- `random`: Sampling aleatorio
- `weighted`: Ponderado por frecuencia de clase
- `site_aware`: Evita data leakage por sitios
- `balanced_batch`: Batches balanceados por clase

## 🎯 Configuraciones Recomendadas por Escenario

### **🏆 Para Competición (Máxima Precisión)**
```bash
python scripts/train_model.py \
    --model efficientnet_b5 \
    --optimizer adamw \
    --learning-rate 0.0008 \
    --weight-decay 0.01 \
    --scheduler cosine \
    --loss focal \
    --class-weights \
    --aggressive-aug \
    --mixed-precision \
    --sampler site_aware \
    --epochs 100 \
    --batch-size 32 \
    --experiment-name "competition_final"
```

### **⚡ Para Desarrollo Rápido**
```bash
python scripts/train_model.py \
    --model efficientnet_b3 \
    --quick-test \
    --fraction 0.2 \
    --epochs 10 \
    --batch-size 32 \
    --experiment-name "development"
```

### **🔬 Para Experimentación Científica**
```bash
python scripts/train_model.py \
    --model resnet101 \
    --optimizer sgd \
    --learning-rate 0.01 \
    --momentum 0.9 \
    --scheduler step \
    --scheduler-step-size 30 \
    --loss cross_entropy \
    --class-weights \
    --wandb \
    --deterministic \
    --experiment-name "scientific_baseline"
```

### **💾 Para Hardware Limitado**
```bash
python scripts/train_model.py \
    --model efficientnet_b0 \
    --batch-size 16 \
    --mixed-precision \
    --num-workers 2 \
    --epochs 50 \
    --experiment-name "low_resource"
```

## 🔍 Monitoreo y Análisis

### **Ver Logs de Entrenamiento**
```bash
# Ver logs en tiempo real
tail -f results/logs/[experiment_name]/training.log

# TensorBoard
tensorboard --logdir results/logs/[experiment_name]/tensorboard

# Weights & Biases (si está habilitado)
# Ir a https://wandb.ai/tu-proyecto
```

### **Resumir desde Checkpoint**
```bash
python scripts/train_model.py \
    --resume results/models/[experiment_name]/checkpoint_epoch_20.pth \
    --epochs 50
```

## 🛠️ Troubleshooting

### **CUDA Out of Memory**
```bash
# Reducir batch size y usar mixed precision
python scripts/train_model.py \
    --batch-size 8 \
    --mixed-precision \
    --num-workers 2
```

### **Data Loading Lento**
```bash
# Usar cache y más workers
python scripts/train_model.py \
    --cache-data \
    --num-workers 8 \
    --pin-memory
```

### **Overfitting**
```bash
# Más regularización
python scripts/train_model.py \
    --dropout 0.7 \
    --weight-decay 0.01 \
    --aggressive-aug \
    --loss label_smoothing \
    --label-smoothing 0.2
```

### **Underfitting**
```bash
# Modelo más grande, menos regularización
python scripts/train_model.py \
    --model efficientnet_b5 \
    --dropout 0.3 \
    --weight-decay 0.0001 \
    --learning-rate 0.001
```

## 📈 Tips de Optimización

1. **Para modelos grandes**: Usa `--mixed-precision` y batch size más pequeño
2. **Para datasets desbalanceados**: Usa `--class-weights` y `--loss focal`
3. **Para máxima precisión**: Combina múltiples modelos después del entrenamiento
4. **Para desarrollo**: Usa `--quick-test` y `--fraction 0.1`
5. **Para reproducibilidad**: Usa `--deterministic` y mismo `--random-state`

## 🎉 Ejemplo Completo de Flujo de Trabajo

```bash
# 1. Test rápido para verificar que todo funciona
python scripts/train_model.py --quick-test --dry-run

# 2. Experimento de desarrollo
python scripts/train_model.py \
    --model efficientnet_b3 \
    --quick-test \
    --fraction 0.1 \
    --epochs 5 \
    --experiment-name "dev_test"

# 3. Entrenamiento completo
python scripts/train_model.py \
    --model efficientnet_b4 \
    --optimizer adamw \
    --learning-rate 0.001 \
    --scheduler cosine \
    --loss focal \
    --class-weights \
    --aggressive-aug \
    --mixed-precision \
    --sampler site_aware \
    --epochs 75 \
    --wandb \
    --experiment-name "final_model"

# 4. Evaluación (usar otro script)
python scripts/evaluate_model.py \
    --model results/models/final_model/best_model.pth \
    --data-dir data/raw
```

¡Este script te da control total sobre todos los aspectos del entrenamiento! 🎯