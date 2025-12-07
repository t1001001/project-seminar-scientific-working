import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

REAL_IMG_DIR = Path("data/preprocessed/images")
REAL_LBL_DIR = Path("data/preprocessed/labels")

SYN_DIR = Path("data/cyclegan/generated")

OUT_DIR = Path("data/yolo/cyclegan_aug")

TRAIN_SPLIT = 0.8
IMG_EXT = ".png"
LBL_EXT = ".txt"

def ensure(p):
    p.mkdir(parents=True, exist_ok=True)

def augment_yolo():
    print("Building Augmented YOLO Dataset...")
    for subset in ["images/train", "images/val", "labels/train", "labels/val"]:
        ensure(OUT_DIR / subset)
    pairs = []
    for scan_folder in REAL_IMG_DIR.iterdir():
        if scan_folder.is_dir():
            for img in scan_folder.glob(f"*{IMG_EXT}"):
                lbl = REAL_LBL_DIR / scan_folder.name / img.name.replace(IMG_EXT, LBL_EXT)
                if lbl.exists():
                    pairs.append((img, lbl))
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print("Copying real data...")
    for img, lbl in tqdm(train_pairs):
        shutil.copy(img, OUT_DIR / "images/train" / img.name)
        shutil.copy(lbl, OUT_DIR / "labels/train" / lbl.name)
    for img, lbl in tqdm(val_pairs):
        shutil.copy(img, OUT_DIR / "images/val" / img.name)
        shutil.copy(lbl, OUT_DIR / "labels/val" / lbl.name)
    print("Copying synthetic images into train/")
    for img in SYN_DIR.glob("*.png"):
        real_lbl = REAL_LBL_DIR.glob(f"*/{img.name.replace('.png','.txt')}")
        real_lbl = list(real_lbl)
        if real_lbl:
            lbl_path = real_lbl[0]
            shutil.copy(img, OUT_DIR / "images/train" / img.name)
            shutil.copy(lbl_path, OUT_DIR / "labels/train" / img.name.replace(".png", ".txt"))
    print("YOLO augmented dataset ready!")

if __name__ == "__main__":
    augment_yolo()
