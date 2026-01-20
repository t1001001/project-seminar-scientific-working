import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import utils.config as conf

PRE_IMG_DIR = Path(f"{conf.ROOT}/data/preprocessed/images")
PRE_LABEL_DIR = Path(f"{conf.ROOT}/data/preprocessed/labels")
YOLO_DATA_DIR = Path(f"{conf.ROOT}/data/yolo/baseline")
SPLIT_PATH = Path(f"{conf.ROOT}/data/split.json")
YAML_PATH = Path(f"{conf.ROOT}/baseline.yaml")

WEIGHTS = "yolo11n.pt"
EPOCHS = 100
IMG_SIZE = 512
BATCH = 32
IMG_EXT = ".png"
LBL_EXT = ".txt"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_split():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError("split.json not found. Run make_split.py first.")
    with open(SPLIT_PATH) as f:
        return json.load(f)

def build_yolo_folders():
    for split in ["train", "val"]:
        ensure_dir(YOLO_DATA_DIR / "images" / split)
        ensure_dir(YOLO_DATA_DIR / "labels" / split)
    print("Done.")

def collect_image_label_pairs():
    pairs = []
    for scan_folder in PRE_IMG_DIR.iterdir():
        if scan_folder.is_dir():
            for img_path in scan_folder.glob(f"*{IMG_EXT}"):
                lbl_path = PRE_LABEL_DIR / scan_folder.name / img_path.name.replace(IMG_EXT, LBL_EXT)
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))
    return pairs

def copy_pairs(pairs, split_name):
    img_out = YOLO_DATA_DIR / "images" / split_name
    lbl_out = YOLO_DATA_DIR / "labels" / split_name
    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, img_out / img_path.name)
        shutil.copy2(lbl_path, lbl_out / lbl_path.name)
    ni = len(list(img_out.glob(f"*{IMG_EXT}")))
    nl = len(list(lbl_out.glob(f"*{LBL_EXT}")))
    print(f"Copied {len(pairs)} files to {split_name}/ (now {ni} imgs, {nl} labels)")

def create_yaml():
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
    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="baseline"
    )

def yolo_baseline():
    split = load_split()
    build_yolo_folders()
    pairs = collect_image_label_pairs()
    train_pairs = []
    val_pairs = []
    for img_path, lbl_path in pairs:
        if img_path.name in split["train"]:
            train_pairs.append((img_path, lbl_path))
        elif img_path.name in split["val"]:
            val_pairs.append((img_path, lbl_path))
    assert len(train_pairs) > 0, "No training samples found!"
    assert len(val_pairs) > 0, "No validation samples found!"
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples: {len(val_pairs)}")
    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "val")
    create_yaml()
    train_yolo()

if __name__ == "__main__":
    yolo_baseline()