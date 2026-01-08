from pathlib import Path

# Paths
DATA_DIR = Path("data/raw")
TRAIN_FEATURES = DATA_DIR / "train_features.csv"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
SUBMISSION_FORMAT = DATA_DIR / "submission_format.csv"
TEST_FEATURES = DATA_DIR / "test_features.csv" # Assuming test features file name if exists, or just use submission format
IMAGE_DIR = DATA_DIR / "images"
PROCESSED_IMAGE_DIR = DATA_DIR / "processed_images"
MODELS_DIR = Path("results/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("results/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64 # From notebook
LEARNING_RATE = 0.01 # Initial LR (check notebook optimizer)
MOMENTUM = 0.909431
WEIGHT_DECAY = 0.005
NUM_EPOCHS = 35 
EARLY_STOPPING_PATIENCE = 5

# Model
NUM_CLASSES = 8
SEED = 42

# Hardware
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
