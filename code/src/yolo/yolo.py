import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import utils.config as conf

PRE_IMG_DIR = Path(f"{conf.ROOT}/data/preprocessed/images")
PRE_LABEL_DIR = Path(f"{conf.ROOT}/data/preprocessed/labels")
YOLO_DATA_DIR = Path(f"{conf.ROOT}/data/yolo/baseline")

TRAIN_SPLIT = 0.8
IMG_EXT = ".png"
LBL_EXT = ".txt"

YAML_PATH = Path(f"{conf.ROOT}/luna.yaml")
WEIGHTS = "yolo11n.pt"
EPOCHS = 50
IMG_SIZE = 512
BATCH = 32

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def build_yolo_folders():
    print("Creating YOLO directory structure...")
    for split in ["train", "val"]:
        ensure_dir(YOLO_DATA_DIR / "images" / split)
        ensure_dir(YOLO_DATA_DIR / "labels" / split)
    print("Done.")

def collect_image_label_pairs():
    print("Collecting image–label pairs...")
    image_paths = []
    for scan_folder in PRE_IMG_DIR.iterdir():
        if scan_folder.is_dir():
            for img_file in scan_folder.glob(f"*{IMG_EXT}"):
                lbl_file = PRE_LABEL_DIR / scan_folder.name / img_file.name.replace(IMG_EXT, LBL_EXT)
                if lbl_file.exists():
                    image_paths.append((img_file, lbl_file))
    print(f"Found {len(image_paths)} slice–label pairs.")
    return image_paths

def split_dataset(pairs):
    print("Splitting into train/val...")
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Training: {len(train_pairs)} | Validation: {len(val_pairs)}")
    return train_pairs, val_pairs

def copy_pairs(pairs, split):
    print(f"Copying {split} set...")
    img_out = YOLO_DATA_DIR / "images" / split
    lbl_out = YOLO_DATA_DIR / "labels" / split
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(lbl_path, lbl_out / lbl_path.name)
    print(f"Copied {len(pairs)} files to {split}/.")

def create_yaml():
    print(f"Creating dataset YAML: {YAML_PATH}")
    ensure_dir(YAML_PATH.parent)
    content = f"""
train: {YOLO_DATA_DIR}/images/train
val: {YOLO_DATA_DIR}/images/val
nc: 1
names: ["nodule"]
"""
    with open(YAML_PATH, "w") as f:
        f.write(content.strip())
    print("YAML created.")

def train_yolo():
    print("Starting YOLOv11 training...")
    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="luna11_baseline"
    )
    print("Training complete!")

def yolo():
    print("YOLO Dataset Builder + Trainer")
    build_yolo_folders()
    pairs = collect_image_label_pairs()
    train_pairs, val_pairs = split_dataset(pairs)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")

    create_yaml()
    train_yolo()

    print("ALL DONE")

if __name__ == "__main__":
    yolo()