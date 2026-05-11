import shutil
import random
from pathlib import Path

# Set Seed for Research Reproducibility
SEED = 42
random.seed(SEED)

def prepare_mango_dataset(source_path: str, output_path: str, split=0.8):
    src_root = Path(source_path)
    dst_root = Path(output_path)
    
    # Automatically detect classes from folder names
    classes = [d.name for d in src_root.iterdir() if d.is_dir()]
    total_processed = 0

    print(f"🚀 Starting dataset split (Seed: {SEED})")

    for label in classes:
        # Define paths
        class_src = src_root / label
        train_dst = dst_root / 'train' / label
        val_dst = dst_root / 'validation' / label

        # Initialize folders
        train_dst.mkdir(parents=True, exist_ok=True)
        val_dst.mkdir(parents=True, exist_ok=True)

        # Collect and shuffle
        files = list(class_src.glob('*.[jJ][pP][gG]')) + list(class_src.glob('*.[pP][nN][gG]'))
        random.shuffle(files)

        split_point = int(len(files) * split)
        
        # Internal helper for clean copying
        def move_set(file_list, target_dir):
            for f in file_list:
                shutil.copy2(f, target_dir / f.name)

        move_set(files[:split_point], train_dst)
        move_set(files[split_point:], val_dst)

        print(f"  ↳ {label.ljust(15)} | Total: {len(files)} | Train: {split_point} | Val: {len(files)-split_point}")
        total_processed += len(files)

    print(f"\n✨ Organization complete. Total images: {total_processed}")

if __name__ == "__main__":
    prepare_mango_dataset(
        source_path='imageformangotree/MangoLeafBD Dataset',
        output_path='D:/opencv/dataset'
    )
