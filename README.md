# Conser-vision Image Classification - Refactored Pipeline

This project is a modular refactoring of an image classification notebook, designed for maintainability and scalability.

## Project Structure

```
.
├── data/
│   └── raw/            # Contains dataset (features, labels, images)
├── src/
│   ├── config.py       # Configuration (paths, hyperparameters)
│   ├── data_loader.py  # Data loading and splitting
│   ├── preprocessing.py# Augmentations and transforms
│   ├── model.py        # ResNet152 model definition
│   ├── train.py        # Training loop
│   ├── predict.py      # Inference and submission logic
│   └── utils.py        # Utilities (early stopping, checkpoints)
├── main.py             # CLI entry point
└── requirements.txt    # Dependencies
```

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The `main.py` script orchestrates the pipeline.

### 1. Verify Data Loading
Check if data can be loaded correctly:
```bash
python main.py --mode test_data
```

### 2. Train the Model
Start the training process. The model with the best validation loss will be saved to `models/best_model.pth`.
```bash
python main.py --mode train
```
*Note: Hyperparameters like batch size and learning rate can be adjusted in `src/config.py`.*

### 3. Generate Submission
Make predictions on the test set and generate `submission.csv`.
```bash
python main.py --mode predict
```

### 4. Dry Run
Run a dry run to test the pipeline with a small subset of data.
```bash
python main.py --mode dry_run
```

## Configuration

Modify `src/config.py` to change:
- **Paths**: Data directories.
- **Hyperparameters**: `BATCH_SIZE`, `LEARNING_RATE`, `NUM_EPOCHS`.
- **Hardware**: The code automatically detects CUDA (GPU) if available.
