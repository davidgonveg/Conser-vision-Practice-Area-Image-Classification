import sys
from pathlib import Path

# Add src to python path to avoid import errors
sys.path.append(str(Path(".").absolute() / "src"))

from src import config
from src.data_loader import load_data, get_data_splits

def verify_split():
    print("Loading data...")
    train_features, train_labels = load_data()
    
    print("Splitting data with GroupShuffleSplit (by site)...")
    groups = train_features["site"]
    x_train, x_eval, y_train, y_eval = get_data_splits(train_features, train_labels, groups=groups)
    
    train_sites = set(train_features.loc[x_train.index, 'site'].unique())
    eval_sites = set(train_features.loc[x_eval.index, 'site'].unique())
    
    intersection = train_sites & eval_sites
    
    print(f"Total sites: {len(groups.unique())}")
    print(f"Train sites: {len(train_sites)}")
    print(f"Eval sites: {len(eval_sites)}")
    
    if len(intersection) == 0:
        print("\nSUCCESS: No site overlap between training and evaluation sets.")
        print("Site-based validation is working correctly.")
        return True
    else:
        print(f"\nFAILURE: Found overlapping sites: {intersection}")
        return False

if __name__ == "__main__":
    success = verify_split()
    if not success:
        sys.exit(1)
