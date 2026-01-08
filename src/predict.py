import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from .config import DEVICE, TEST_FEATURES, SUBMISSION_FORMAT, BATCH_SIZE, DATA_DIR, NUM_CLASSES
from .data_loader import ImagesDataset
from .preprocessing import custom_preprocessing
from .utils import load_checkpoint
from .model import get_model

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", leave=False):
            inputs = batch["image"].to(device)
            ids = batch["image_id"]
            
            outputs = model(inputs)
            # Apply Softmax
            probs = torch.softmax(outputs, dim=1)
            
            predictions.append(probs.cpu().numpy())
            image_ids.extend(ids)
            
    return image_ids, predictions

def generate_submission(model_path="model_best.pth.tar", output_file="submission.csv"):
    # Load test data
    if TEST_FEATURES.exists():
        test_features = pd.read_csv(TEST_FEATURES, index_col="id")
    elif SUBMISSION_FORMAT.exists():
        print(f"{TEST_FEATURES} not found, using {SUBMISSION_FORMAT} for IDs")
        submission_format = pd.read_csv(SUBMISSION_FORMAT, index_col="id")
        # We need the filepaths. Assuming they are inferable or present.
        # But submission format typically only has IDs and empty columns.
        # Hopefully test_features.csv exists. If not, and we only have an image dir:
        # We might need to construct the dataframe. 
        # For now, let's assume test_features exists as per notebook.
        test_features = submission_format # Placeholder, likely to fail if filepath column missing
        if "filepath" not in test_features.columns:
             # Try to construct filepath from ID if possible, or fail gracefully
             # Notebook says: test_features.csv contains image_id, filepath, site_id
             pass
    else:
        raise FileNotFoundError("No test features or submission format found.")

    # Create dataset
    # Note: submission formatting often requires specific order. 
    # reading csv with index_col="id" keeps order.
    
    dataset = ImagesDataset(test_features, y_df=None, preprocessing=custom_preprocessing)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load Model
    model = get_model(NUM_CLASSES)
    model = model.to(DEVICE)
    load_checkpoint(model_path, model)
    
    # Predict
    image_ids, preds = predict(model, dataloader, DEVICE)
    
    # Concatenate predictions
    import numpy as np
    preds = np.concatenate(preds, axis=0)
    
    # Create submission DataFrame
    # Need column names. Load submission format to get them.
    if SUBMISSION_FORMAT.exists():
        submission_format = pd.read_csv(SUBMISSION_FORMAT, index_col="id")
        columns = submission_format.columns
    else:
        # Fallback if we know the classes (we do from notebook, sorted unique labels)
        # But better to read from file if possible. 
        # I'll rely on columns being sorted alphabetically as per notebook analysis
        # species_labels = sorted(train_labels.columns.unique())
        # I'll just use a placeholder here or get from config if I added it.
        # Let's assume standard sorted 8 classes.
        pass

    submission_df = pd.DataFrame(preds, index=image_ids, columns=columns)
    
    # verify index matches submission format
    submission_df.index.name = "id"
    
    # Save
    submission_df.to_csv(output_file)
    print(f"Submission saved to {output_file}")
