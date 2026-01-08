import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
from .config import IMAGE_DIR, DATA_DIR

# Define cache directory
CACHE_DIR = DATA_DIR / "processed_images"

def resize_and_save(args):
    """
    Helper function for parallel processing.
    args: (source_path, target_path, size)
    """
    source_path, target_path, size = args
    try:
        with Image.open(source_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.Resampling.BILINEAR)
            img.save(target_path, "JPEG", quality=90)
    except Exception as e:
        print(f"Error processing {source_path}: {e}")

def prepare_cached_dataset(target_size=(224, 224), clear_cache=True):
    """
    Resizes all images from IMAGE_DIR to CACHE_DIR.
    """
    print(f"Preparing cached dataset at {CACHE_DIR}...")
    
    if clear_cache and CACHE_DIR.exists():
        print("Clearing existing cache...")
        shutil.rmtree(CACHE_DIR)
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # List all images
    # Assuming shallow structure as per current data loader logic
    image_files = list(IMAGE_DIR.glob("*.jpg")) + list(IMAGE_DIR.glob("*.png")) + list(IMAGE_DIR.glob("*.jpeg"))
    
    if not image_files:
        print(f"No images found in {IMAGE_DIR}")
        return CACHE_DIR

    tasks = []
    for img_path in image_files:
        target_path = CACHE_DIR / img_path.name
        tasks.append((img_path, target_path, target_size))
    
    # Use roughly #CPUs - 1 workers
    workers = max(1, os.cpu_count() - 1)
    
    print(f"Resizing {len(image_files)} images using {workers} workers...")
    
    # Using ProcessPoolExecutor for parallel resizing
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(resize_and_save, tasks), total=len(tasks), unit="img"))
        
    print("Dataset caching complete.")
    return CACHE_DIR
