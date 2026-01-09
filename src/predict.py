import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from .config import DEVICE, TEST_FEATURES, SUBMISSION_FORMAT, BATCH_SIZE, DATA_DIR, NUM_CLASSES, MODELS_DIR, SUBMISSIONS_DIR, N_FOLDS
from .data_loader import ImagesDataset
from .preprocessing import custom_preprocessing
from .utils import load_checkpoint
from .model import get_model

def predict(model, dataloader, device):
    model.eval()
    predictions = []
    image_ids = []
    
    with torch.no_grad():
        # Use AMP for inference speedup
        with torch.amp.autocast('cuda',):
            for batch in tqdm(dataloader, desc="Predicting", leave=False):
                inputs = batch["image"].to(device)
                ids = batch["image_id"]
                
                outputs = model(inputs)
                # Apply Softmax
                probs = torch.softmax(outputs, dim=1)
                
                predictions.append(probs.cpu().numpy())
                image_ids.extend(ids)
            
    return image_ids, predictions

def generate_submission(model_filename="model_best.pth.tar", output_filename="submission.csv"):
    # Output paths
    model_path = MODELS_DIR / model_filename
    output_file = SUBMISSIONS_DIR / output_filename

    # Load test data
    if TEST_FEATURES.exists():
        test_features = pd.read_csv(TEST_FEATURES, index_col="id")
    elif SUBMISSION_FORMAT.exists():
        print(f"{TEST_FEATURES} not found, using {SUBMISSION_FORMAT} for IDs")
        submission_format = pd.read_csv(SUBMISSION_FORMAT, index_col="id")
        test_features = submission_format
        if "filepath" not in test_features.columns:
             pass
    else:
        raise FileNotFoundError("No test features or submission format found.")

    # Create dataset
    # Optimization: num_workers=4, pin_memory=True
    dataset = ImagesDataset(test_features, y_df=None, preprocessing=custom_preprocessing)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load Model
    model = get_model(NUM_CLASSES)
    # Ensemble Loop
    ensemble_preds = []
    
    # Check if we have fold models available
    fold_models = list(MODELS_DIR.glob("model_fold_*.pth.tar"))
    
    if len(fold_models) > 0:
        print(f"Found {len(fold_models)} fold models for ensembling.")
        models_to_run = fold_models
    else:
        # Fallback to single best model
        print("No fold models found. Falling back to {model_filename}")
        if not model_path.exists():
             print(f"Model not found at {model_path}. Trying root...")
             if Path(model_filename).exists():
                 model_path = Path(model_filename)
        models_to_run = [model_path]

    # Iterate over models
    import numpy as np
    
    final_preds_accum = None
    final_image_ids = None

    for m_path in models_to_run:
        print(f"Predicting with {m_path.name}...")
        current_model = get_model(NUM_CLASSES)
        current_model = current_model.to(DEVICE)
        load_checkpoint(m_path, current_model)
        
        ids, preds = predict(current_model, dataloader, DEVICE)
        # preds is list of arrays, concat
        preds_array = np.concatenate(preds, axis=0)
        
        if final_preds_accum is None:
            final_preds_accum = preds_array
            final_image_ids = ids
        else:
            final_preds_accum += preds_array
            
    # Average predictions
    avg_preds = final_preds_accum / len(models_to_run)
    preds = avg_preds 
    image_ids = final_image_ids

    
    # Create submission DataFrame
    if SUBMISSION_FORMAT.exists():
        submission_format = pd.read_csv(SUBMISSION_FORMAT, index_col="id")
        columns = submission_format.columns
    else:
        # Fallback
        # Assuming we can get classes from config or sorted list if strictly necessary
        pass

    submission_df = pd.DataFrame(preds, index=image_ids, columns=columns)
    
    # verify index matches submission format
    submission_df.index.name = "id"
    
    # Save
    submission_df.to_csv(output_file)
    print(f"Submission saved to {output_file}")
