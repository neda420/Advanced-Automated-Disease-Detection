import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = Path('imageformangotree/MangoLeafBD Dataset')
BASE_DIR = Path('D:/opencv/dataset')
TRAIN_RATIO = 0.8

disease_classes = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 
    'Die Back', 'Gall Midge', 'Powdery Mildew', 
    'Sooty Mould', 'Healthy'
]

def organize_dataset():
    for cls in disease_classes:
        src_path = SOURCE_DIR / cls
        train_path = BASE_DIR / 'train' / cls
        val_path = BASE_DIR / 'validation' / cls

        # Ensure directories exist
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        if not src_path.exists():
            print(f"⚠️ Skipping: {cls} (Source not found)")
            continue

        # Filter for actual image files
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Process copies
        for img in train_files:
            shutil.copy2(src_path / img, train_path / img)
        for img in val_files:
            shutil.copy2(src_path / img, val_path / img)

        print(f"✅ {cls}: {len(train_files)} train, {len(val_files)} val")

if __name__ == "__main__":
    organize_dataset()
