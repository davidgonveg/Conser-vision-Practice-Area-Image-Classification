# Taï National Park Camera Trap Species Classification

This project implements a deep learning solution for classifying species in camera trap images from Taï National Park. The goal is to help conservation efforts by automatically identifying different animal species in camera trap footage.

## Project Structure

```
tai-park-species-classification/
├── data/                    # Data directory
│   ├── raw/                # Original data files
│   ├── processed/          # Processed data
│   └── submissions/        # Generated submissions
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model definitions
│   ├── training/          # Training logic
│   ├── evaluation/        # Evaluation and metrics
│   ├── inference/         # Inference and prediction
│   └── utils/             # Utility functions
├── scripts/               # Executable scripts
├── notebooks/             # Jupyter notebooks for exploration
├── configs/               # Configuration files
├── results/               # Results, models, and logs
└── tests/                 # Unit tests
```

## Species Classes

The model classifies images into 8 categories:
- antelope_duiker
- bird
- blank (no animals)
- civet_genet
- hog
- leopard
- monkey_prosimian
- rodent

## Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate tai-park-species
```

### Using pip

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e .
```

## Usage

### Data Preparation

1. Place the competition data in `data/raw/`:
   - `train_features/` - Training images
   - `test_features/` - Test images
   - `train_features.csv` - Training metadata
   - `test_features.csv` - Test metadata
   - `train_labels.csv` - Training labels
   - `submission_format.csv` - Submission format

### Training

```bash
python scripts/train_model.py --config configs/base_config.yaml
```

### Evaluation

```bash
python scripts/evaluate_model.py --model results/models/best_model.pth
```

### Generate Submission

```bash
python scripts/generate_submission.py --model results/models/best_model.pth
```

## Key Features

- **Site-aware validation**: Ensures models generalize to new camera trap sites
- **Image augmentation**: Robust preprocessing for varied lighting and angles
- **Multiple model architectures**: Support for various CNN architectures
- **Ensemble methods**: Combine multiple models for better performance
- **Comprehensive evaluation**: Detailed metrics and visualization tools

## Competition Details

This project is designed for the Taï National Park camera trap species classification competition. The evaluation metric is log loss, and the challenge emphasizes model generalization to unseen camera trap sites.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

## License

This project is licensed under the MIT License.
