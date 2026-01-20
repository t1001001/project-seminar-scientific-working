import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import utils.config as conf
from concurrent.futures import ThreadPoolExecutor

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
        split = json.load(f)
    return set(split["train"]), set(split["val"])

def build_yolo_folders():
    for split in ["train", "val"]:
        ensure_dir(YOLO_DATA_DIR / "images" / split)
        ensure_dir(YOLO_DATA_DIR / "labels" / split)

def collect_image_label_pairs():
    lbl_index = {p.name: p for p in PRE_LABEL_DIR.rglob(f"*{LBL_EXT}")}
    pairs = []
    for img_path in PRE_IMG_DIR.rglob(f"*{IMG_EXT}"):
        lbl_path = lbl_index.get(img_path.name.replace(IMG_EXT, LBL_EXT))
        if lbl_path:
            pairs.append((img_path, lbl_path))
    return pairs

def copy_pairs_parallel(pairs, split_name):
    img_out = YOLO_DATA_DIR / "images" / split_name
    lbl_out = YOLO_DATA_DIR / "labels" / split_name
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(lambda p: (shutil.copy2(p[0], img_out / p[0].name),
                          shutil.copy2(p[1], lbl_out / p[1].name)), pairs)
    print(f"Copied {len(pairs)} files to {split_name}/")

def create_yaml():
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
    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="baseline",
        workers=8
    )

def yolo_baseline():
    train_set, val_set = load_split()
    build_yolo_folders()
    pairs = collect_image_label_pairs()
    train_pairs = [p for p in pairs if p[0].name in train_set]
    val_pairs = [p for p in pairs if p[0].name in val_set]
    assert train_pairs, "No training samples found!"
    assert val_pairs, "No validation samples found!"
    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples: {len(val_pairs)}")
    copy_pairs_parallel(train_pairs, "train")
    copy_pairs_parallel(val_pairs, "val")
    create_yaml()
    train_yolo()

if __name__ == "__main__":
    yolo_baseline()