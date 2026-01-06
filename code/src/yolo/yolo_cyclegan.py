import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import utils.config as conf

PRE_IMG_DIR = Path(f"{conf.ROOT}/data/preprocessed/images")
PRE_LABEL_DIR = Path(f"{conf.ROOT}/data/preprocessed/labels")
SYN_DIR = Path(f"{conf.ROOT}/data/cyclegan/generated")
YOLO_DATA_DIR = Path(f"{conf.ROOT}/data/yolo/cyclegan")
SPLIT_PATH = Path(f"{conf.ROOT}/data/split.json")
YAML_PATH = Path(f"{conf.ROOT}/cyclegan.yaml")

WEIGHTS = "yolo11n.pt"
EPOCHS = 50
IMG_SIZE = 512
BATCH = 32
IMG_EXT = ".png"
LBL_EXT = ".txt"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def prepare_dataset():
    split = load_split()
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    for name in split["train"]:
        for scan_folder in PRE_IMG_DIR.iterdir():
            img_file = scan_folder / name
            lbl_file = PRE_LABEL_DIR / scan_folder.name / name.replace(".png", ".txt")
            if img_file.exists() and lbl_file.exists():
                shutil.copy(img_file, YOLO_DATA_DIR / "images/train" / name)
                shutil.copy(lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(".png", ".txt"))
                break
        syn_file = SYN_DIR / name
        if syn_file.exists():
            out_name = f"cyc_{name}"
            shutil.copy(syn_file, YOLO_DATA_DIR / "images/train" / out_name)
            shutil.copy(lbl_file, YOLO_DATA_DIR / "labels/train" / out_name.replace(".png", ".txt"))
    for name in split["val"]:
        for scan_folder in PRE_IMG_DIR.iterdir():
            img_file = scan_folder / name
            lbl_file = PRE_LABEL_DIR / scan_folder.name / name.replace(".png", ".txt")
            if img_file.exists() and lbl_file.exists():
                shutil.copy(img_file, YOLO_DATA_DIR / "images/val" / name)
                shutil.copy(lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(".png", ".txt"))
                break

    print("Dataset prepared according to split.json")

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

def yolo_cyclegan():
    print("\n=== YOLO CycleGAN (Experiment C) ===")
    prepare_dataset()
    create_yaml()

    print("Training YOLOv11 on CycleGAN-Augmented Dataset")
    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="cyclegan"
    )
    print("=== EXPERIMENT C DONE ===\n")

if __name__ == "__main__":
    yolo_cyclegan()