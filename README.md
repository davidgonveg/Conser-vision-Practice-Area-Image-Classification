# Taï National Park Camera Trap Species Classification

A comprehensive deep learning pipeline for classifying wildlife species in camera trap images from Taï National Park. This project helps conservation researchers automatically identify different animal species captured by camera traps, reducing manual effort and enabling large-scale wildlife monitoring.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for classifying camera trap images into 8 categories:
- **7 animal species**: `antelope_duiker`, `bird`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, `rodent`
- **1 blank category**: `blank` (no animals present)

**Key Challenge**: The model must generalize to new camera trap sites (locations) that weren't seen during training, making this a domain adaptation problem.

## 🏗️ Project Structure

```
tai-park-species-classification/
├── data/                           # Data directory
│   ├── raw/                       # Original competition data
│   │   ├── train_features/        # Training images
│   │   ├── test_features/         # Test images
│   │   ├── train_features.csv     # Training metadata
│   │   ├── test_features.csv      # Test metadata
│   │   └── train_labels.csv       # Training labels
│   ├── processed/                 # Preprocessed data
│   │   └── validation_sites.csv   # Validation sites list
│   └── submissions/               # Generated submissions
├── src/                           # Source code
│   ├── data/                      # Data handling modules
│   │   ├── dataset.py            # PyTorch dataset classes
│   │   ├── transforms.py         # Image transformations
│   │   ├── data_loader.py        # Data loaders and samplers
│   │   └── preprocessing.py      # Data preprocessing utilities
│   ├── models/                    # Model architectures
│   │   └── model.py              # Model definitions
│   ├── training/                  # Training modules
│   │   ├── trainer.py            # Training loop
│   │   └── losses.py             # Loss functions
│   ├── evaluation/                # Evaluation modules
│   │   └── metrics.py            # Evaluation metrics
│   ├── inference/                 # Inference modules
│   │   ├── predictor.py          # Prediction pipeline
│   │   └── tta.py                # Test time augmentation
│   └── utils/                     # Utility modules
│       ├── config.py             # Configuration management
│       └── logging_utils.py      # Logging utilities
├── scripts/                       # Executable scripts
│   ├── train_model.py            # Training script
│   ├── evaluate_model.py         # Evaluation script
│   └── generate_submission.py    # Submission generation
├── configs/                       # Configuration files
│   └── base_config.yaml          # Base configuration
├── notebooks/                     # Jupyter notebooks (for exploration)
├── results/                       # Training results
│   ├── models/                   # Saved models
│   ├── logs/                     # Training logs
│   └── plots/                    # Generated plots
└── tests/                        # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

**Option A: Using Conda (Recommended)**
```bash
# Create environment
conda env create -f environment.yml
conda activate tai-park-species

# Install project in development mode
pip install -e .
```

**Option B: Using pip**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Data Preparation

1. **Download competition data** and place it in `data/raw/`:
   ```
   data/raw/
   ├── train_features/        # Training images
   ├── test_features/         # Test images
   ├── train_features.csv     # Training metadata
   ├── test_features.csv      # Test metadata
   ├── train_labels.csv       # Training labels
   └── submission_format.csv  # Submission format
   ```

2. **Quick data validation** (optional):
   ```bash
   python -c "from src.data.preprocessing import analyze_dataset_quick; analyze_dataset_quick('data/raw')"
   ```

### 3. Training a Model

**Basic training:**
```bash
python scripts/train_model.py
```

**Advanced training with custom settings:**
```bash
python scripts/train_model.py \
    --model efficientnet_b3 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --aggressive-aug \
    --class-weights \
    --sampler site_aware \
    --experiment-name "efficientnet_b3_experiment"
```

**Quick test run:**
```bash
python scripts/train_model.py --quick-test --dry-run
```

### 4. Evaluating a Model

```bash
python scripts/evaluate_model.py \
    --model results/models/best_model.pth \
    --data-dir data/raw \
    --save-plots \
    --detailed-analysis
```

### 5. Generating Submissions

**Single model submission:**
```bash
python scripts/generate_submission.py \
    --model results/models/best_model.pth \
    --data-dir data/raw \
    --output submissions/my_submission.csv
```

**Ensemble submission with TTA:**
```bash
python scripts/generate_submission.py \
    --ensemble results/models/model1.pth results/models/model2.pth \
    --use-tta \
    --output submissions/ensemble_tta_submission.csv
```

## 📋 Detailed Module Documentation

### 🗂️ Data Module (`src/data/`)

The data module handles all aspects of data loading, preprocessing, and augmentation:

#### **`dataset.py`** - Core Dataset Classes
- `TaiParkDataset`: Main dataset class with site-based validation splits
- `create_datasets()`: Factory function for train/val datasets
- `create_test_dataset()`: Factory function for test dataset

```python
from src.data import create_datasets, get_train_transforms, get_val_transforms

# Create datasets
train_transform = get_train_transforms(image_size=224)
val_transform = get_val_transforms(image_size=224)

train_ds, val_ds = create_datasets(
    data_dir="data/raw",
    train_transform=train_transform,
    val_transform=val_transform
)
```

#### **`transforms.py`** - Image Transformations
Wildlife-specific image transformations that handle camera trap challenges:

- **`AdaptiveBrightnessContrast`**: Handles day/night variations
- **`WildlifeSpecificRotation`**: Animal-appropriate rotations
- **`CameraTrapCrop`**: Smart cropping for different animal distances
- **`AnimalFriendlyFlip`**: Horizontal flips (no vertical flips)
- **`EnvironmentalNoise`**: Simulates weather conditions

```python
from src.data.transforms import get_train_transforms

# Get training transforms
train_transform = get_train_transforms(
    image_size=224,
    aggressive=True  # More aggressive augmentation
)
```

#### **`data_loader.py`** - Data Loaders and Samplers
Specialized data loaders for camera trap data:

- **`DataLoaderManager`**: Easy configuration for all data loading
- **`SiteAwareBatchSampler`**: Ensures site diversity in batches
- **`BalancedBatchSampler`**: Balanced class representation
- **`wildlife_collate_fn`**: Custom collate function with metadata

```python
from src.data import DataLoaderManager

# Quick setup
manager = DataLoaderManager(
    data_dir="data/raw",
    batch_size=32,
    train_sampler_type="site_aware",
    aggressive_augmentation=False
)

train_loader = manager.train_loader
val_loader = manager.val_loader
```

#### **`preprocessing.py`** - Data Analysis and Cleaning
Comprehensive data preprocessing and analysis:

- **`DatasetAnalyzer`**: Analyzes dataset structure and quality
- **`ImageValidator`**: Validates image files
- **`CacheManager`**: Manages data caching
- **`preprocess_dataset()`**: Batch image preprocessing

```python
from src.data.preprocessing import DatasetAnalyzer

# Analyze dataset
analyzer = DatasetAnalyzer("data/raw")
report = analyzer.generate_report()
print(f"Dataset health: {report['summary']['dataset_health']}")
```

### 🧠 Models Module (`src/models/`)

#### **`model.py`** - Model Architectures
Flexible model architecture with multiple options:

- **`WildlifeClassifier`**: Main classifier with customizable backbone
- **`SpatialAttention`**: Attention mechanism for focusing on animals
- **`EnsembleClassifier`**: Combines multiple models
- **`MultiScaleClassifier`**: Handles different image scales

```python
from src.models.model import create_model

# Create model
model = create_model(
    model_name="efficientnet_b3",
    num_classes=8,
    pretrained=True,
    dropout=0.3,
    use_site_embedding=True  # For domain adaptation
)
```

**Available model architectures:**
- `efficientnet_b0` to `efficientnet_b7`
- `resnet50`, `resnet101`, `resnet152`
- `convnext_base`, `convnext_large`
- `vit_base_patch16_224`

**Predefined configurations:**
```python
from src.models.model import get_model_config

# Use predefined config
config = get_model_config("site_aware")  # baseline, efficient, site_aware, large
model = create_model(**config)
```

### 🎯 Training Module (`src/training/`)

#### **`trainer.py`** - Training Pipeline
Comprehensive training with advanced features:

- **Mixed precision training** for faster training on modern GPUs
- **Site-aware validation** to ensure generalization
- **Advanced logging** with TensorBoard and Weights & Biases
- **Automatic checkpointing** and early stopping

```python
from src.training.trainer import Trainer
from src.training.losses import get_loss_function

# Create trainer
trainer = Trainer(
    model=model,
    criterion=get_loss_function("focal", class_weights=class_weights),
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    config=config,
    output_dir=output_dir,
    logger=logger,
    mixed_precision=True
)

# Train
trainer.train(train_loader, val_loader, num_epochs=50)
```

#### **`losses.py`** - Loss Functions
Specialized loss functions for imbalanced wildlife data:

- **`FocalLoss`**: Handles class imbalance
- **`LabelSmoothingCrossEntropy`**: Improves generalization
- **`ClassBalancedLoss`**: Based on effective number of samples
- **`OnlineHardExampleMining`**: Focuses on difficult examples

```python
from src.training.losses import get_loss_function, compute_class_weights

# Compute class weights
class_weights = compute_class_weights(class_counts, method='inverse_frequency')

# Create loss function
criterion = get_loss_function("focal", class_weights=class_weights, gamma=2.0)
```

### 📊 Evaluation Module (`src/evaluation/`)

#### **`metrics.py`** - Evaluation Metrics
Comprehensive evaluation metrics:

- **`MetricsCalculator`**: Calculates all metrics
- **Log loss** (competition metric)
- **Class-wise precision, recall, F1-score**
- **Site-wise performance analysis**
- **Confusion matrix analysis**

```python
from src.evaluation.metrics import MetricsCalculator

# Calculate metrics
metrics_calc = MetricsCalculator(num_classes=8)
metrics = metrics_calc.calculate_all_metrics(
    y_true=labels,
    y_pred=predictions,
    y_proba=probabilities
)

print(f"Log Loss: {metrics['log_loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### 🔮 Inference Module (`src/inference/`)

#### **`predictor.py`** - Prediction Pipeline
Production-ready prediction pipeline:

- **`WildlifePredictor`**: Main prediction class
- **Batch processing** for efficiency
- **Test Time Augmentation** support
- **Automatic submission generation**

```python
from src.inference.predictor import WildlifePredictor

# Create predictor
predictor = WildlifePredictor(
    model_path="results/models/best_model.pth",
    device="cuda",
    use_tta=True,
    batch_size=32
)

# Single image prediction
result = predictor.predict_single("path/to/image.jpg")
print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.3f})")

# Generate submission
submission = predictor.generate_submission(
    test_features_path="data/raw/test_features.csv",
    test_images_dir="data/raw/test_features",
    output_path="submissions/my_submission.csv"
)
```

#### **`tta.py`** - Test Time Augmentation
Advanced TTA strategies:

- **Multiple augmentation strategies**: flip, rotate, crop, color
- **Wildlife-specific augmentations**
- **Configurable averaging methods**

```python
from src.inference.tta import create_tta_config, TTAPredictor

# Create TTA configuration
tta_config = create_tta_config(
    strategy="comprehensive",
    n_augmentations=8,
    image_size=224
)

# Use TTA predictor
tta_predictor = TTAPredictor(model, tta_config, device)
result = tta_predictor.predict(image)
```

## 🎛️ Configuration System

The project uses YAML configuration files for easy parameter management:

### **`configs/base_config.yaml`** - Main Configuration
```yaml
# Model settings
model:
  name: "efficientnet_b3"
  num_classes: 8
  pretrained: true
  dropout: 0.3

# Training settings
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 10

# Data settings
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

# Image settings
image:
  size: [224, 224]
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

### Using Configuration in Code
```python
from src.utils.config import Config

# Load configuration
config = Config("configs/base_config.yaml")

# Access values
batch_size = config.get('training.batch_size')
model_name = config.get('model.name')
```

## 🛠️ Command Line Scripts

### **`scripts/train_model.py`** - Training Script

**Basic usage:**
```bash
python scripts/train_model.py --config configs/base_config.yaml
```

**Advanced options:**
```bash
python scripts/train_model.py \
    --model efficientnet_b4 \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --weight-decay 0.0001 \
    --aggressive-aug \
    --class-weights \
    --focal-loss \
    --sampler site_aware \
    --mixed-precision \
    --wandb \
    --experiment-name "efficientnet_b4_focal_loss"
```

**Quick testing:**
```bash
python scripts/train_model.py --quick-test --dry-run
```

### **`scripts/evaluate_model.py`** - Evaluation Script

**Basic evaluation:**
```bash
python scripts/evaluate_model.py \
    --model results/models/best_model.pth \
    --data-dir data/raw
```

**Detailed evaluation with visualizations:**
```bash
python scripts/evaluate_model.py \
    --model results/models/best_model.pth \
    --data-dir data/raw \
    --save-predictions \
    --save-plots \
    --detailed-analysis \
    --use-tta
```

### **`scripts/generate_submission.py`** - Submission Generation

**Single model submission:**
```bash
python scripts/generate_submission.py \
    --model results/models/best_model.pth \
    --data-dir data/raw \
    --output submissions/single_model.csv
```

**Ensemble submission:**
```bash
python scripts/generate_submission.py \
    --ensemble results/models/model1.pth results/models/model2.pth results/models/model3.pth \
    --ensemble-weights 0.4 0.4 0.2 \
    --use-tta \
    --output submissions/ensemble_tta.csv
```

## 📈 Advanced Training Strategies

### 1. **Site-Aware Training**
Ensures models generalize to new camera trap locations:

```bash
python scripts/train_model.py \
    --sampler site_aware \
    --validation-sites data/processed/validation_sites.csv
```

### 2. **Class Balancing**
Handles imbalanced wildlife data:

```bash
python scripts/train_model.py \
    --class-weights \
    --focal-loss \
    --sampler balanced_batch
```

### 3. **Advanced Augmentation**
Wildlife-specific data augmentation:

```bash
python scripts/train_model.py \
    --aggressive-aug \
    --model efficientnet_b4
```

### 4. **Multi-GPU Training**
For faster training on multiple GPUs:

```bash
python scripts/train_model.py \
    --mixed-precision \
    --batch-size 128 \
    --num-workers 8
```

### 5. **Experiment Tracking**
With Weights & Biases:

```bash
python scripts/train_model.py \
    --wandb \
    --wandb-project "tai-park-species" \
    --experiment-name "efficientnet_b5_experiment"
```

## 🔧 Development and Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_dataset.py

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/
```

### Data Validation
```bash
# Quick dataset analysis
python -c "from src.data.preprocessing import analyze_dataset_quick; analyze_dataset_quick()"

# Detailed dataset analysis
python -c "from src.data.preprocessing import DatasetAnalyzer; analyzer = DatasetAnalyzer('data/raw'); report = analyzer.generate_report()"
```

## 🏆 Competition Tips

### 1. **Model Selection**
- Start with `efficientnet_b3` for good balance of performance and speed
- Use `efficientnet_b5` or `efficientnet_b7` for maximum accuracy
- Consider `convnext_base` for state-of-the-art performance

### 2. **Training Strategy**
- Always use site-aware validation (`--sampler site_aware`)
- Use class weights for imbalanced data (`--class-weights`)
- Consider focal loss for difficult classes (`--focal-loss`)
- Enable mixed precision for faster training (`--mixed-precision`)

### 3. **Ensembling**
- Train multiple models with different architectures
- Use different augmentation strategies
- Combine with Test Time Augmentation (`--use-tta`)

### 4. **Submission Best Practices**
- Validate submission format automatically
- Use TTA for final submissions
- Ensemble 3-5 diverse models
- Save detailed probabilities for analysis

## 📚 Examples and Tutorials

### Example 1: Basic Training Pipeline
```python
from src.data import get_quick_setup
from src.models.model import create_model
from src.training.trainer import Trainer

# Quick setup
manager = get_quick_setup(
    data_dir="data/raw",
    batch_size=32,
    sampler_type="site_aware"
)

# Create model
model = create_model("efficientnet_b3", pretrained=True)

# Train (simplified)
# ... create optimizer, scheduler, trainer ...
# trainer.train(manager.train_loader, manager.val_loader, num_epochs=50)
```

### Example 2: Custom Data Processing
```python
from src.data import TaiParkDataset, get_train_transforms
from torch.utils.data import DataLoader

# Create custom dataset
transform = get_train_transforms(image_size=224, aggressive=True)
dataset = TaiParkDataset(
    data_dir="data/raw",
    split="train",
    transform=transform,
    return_site_info=True
)

# Create dataloader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Example 3: Model Evaluation
```python
from src.inference.predictor import WildlifePredictor
from src.evaluation.metrics import MetricsCalculator

# Load predictor
predictor = WildlifePredictor("results/models/best_model.pth")

# Evaluate
metrics_calc = MetricsCalculator(num_classes=8)
# ... get predictions and calculate metrics ...
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/train_model.py --batch-size 16
   
   # Use mixed precision
   python scripts/train_model.py --mixed-precision
   ```

2. **Data Loading Errors**
   ```bash
   # Check data structure
   python -c "from src.data.preprocessing import analyze_dataset_quick; analyze_dataset_quick()"
   
   # Reduce number of workers
   python scripts/train_model.py --num-workers 2
   ```

3. **Import Errors**
   ```bash
   # Install in development mode
   pip install -e .
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Performance Tips

1. **Speed up training:**
   - Use `--mixed-precision`
   - Increase `--batch-size` if GPU memory allows
   - Use `--compile` (PyTorch 2.0+)

2. **Improve accuracy:**
   - Use larger models (`efficientnet_b5`, `efficientnet_b7`)
   - Enable `--aggressive-aug`
   - Use `--class-weights` and `--focal-loss`
   - Ensemble multiple models

3. **Save GPU memory:**
   - Reduce `--batch-size`
   - Use `--mixed-precision`
   - Reduce `--num-workers`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Wild Chimpanzee Foundation** and **Max Planck Institute for Evolutionary Anthropology** for providing the dataset
- **Taï National Park** for conservation efforts
- The open-source community for the underlying frameworks and libraries

---

**Happy predicting! 🐆📸**

For more detailed documentation, check the individual module docstrings and example notebooks in the `notebooks/` directory.