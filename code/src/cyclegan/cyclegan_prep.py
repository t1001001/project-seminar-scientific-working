import os
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance
from tqdm import tqdm
import utils.config as conf

SOURCE_DIR = Path(f"{conf.ROOT}/data/preprocessed/images")
CYCLEGAN_DIR = Path(f"{conf.ROOT}/data/cyclegan")
TRAIN_SPLIT = 0.8

def ensure(path):
    path.mkdir(parents=True, exist_ok=True)

def style_transform(image_path):
    img = Image.open(image_path).convert("L")
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Brightness(img).enhance(1.15)
    return img

def cyclegan_prep():
    print("Preparing CycleGAN datasets...")
    for sub in ["trainA", "trainB", "testA", "testB"]:
        ensure(CYCLEGAN_DIR / sub)
    all_slices = []
    for scan_folder in SOURCE_DIR.iterdir():
        if scan_folder.is_dir():
            all_slices.extend(scan_folder.glob("*.png"))
    print(f"Found {len(all_slices)} slices.")
    random.seed(42)
    all_slices = list(all_slices)
    random.shuffle(all_slices)
    split_idx = int(len(all_slices) * TRAIN_SPLIT)
    train_slices = all_slices[:split_idx]
    test_slices = all_slices[split_idx:]
    for img_path in tqdm(train_slices):
        shutil.copy2(img_path, CYCLEGAN_DIR / "trainA" / img_path.name)
        styled = style_transform(img_path)
        styled.save(CYCLEGAN_DIR / "trainB" / img_path.name)
    for img_path in tqdm(test_slices):
        shutil.copy2(img_path, CYCLEGAN_DIR / "testA" / img_path.name)
        styled = simple_style_transform(img_path)
        styled.save(CYCLEGAN_DIR / "testB" / img_path.name)
    print("CycleGAN data created!")

if __name__ == "__main__":
    cyclegan_prep()
