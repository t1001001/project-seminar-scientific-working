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
EPOCHS = 100
IMG_SIZE = 512
BATCH = 32
IMG_EXT = ".png"
LBL_EXT = ".txt"

# CycleGAN suffixes
SUFFIXES = ["_fake_A", "_fake_B", "_real_A", "_real_B", "_rec_A", "_rec_B"]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def find_image_by_name(name: str) -> Path | None:
    matches = list(PRE_IMG_DIR.glob(f"*/{name}"))
    return matches[0] if matches else None

def find_label_by_name(name: str) -> Path | None:
    matches = list(PRE_LABEL_DIR.glob(f"*/{name.replace(IMG_EXT, LBL_EXT)}"))
    return matches[0] if matches else None

def find_syn_by_name(name: str) -> Path | None:
    stem = Path(name).stem
    candidates = list(SYN_DIR.glob(f"{stem}_*.png"))
    if not candidates:
        return None

    def rank(p: Path):
        s = p.stem
        for i, suf in enumerate(SUFFIXES):
            if s.endswith(suf):
                return i
        return len(SUFFIXES)

    candidates.sort(key=rank)
    return candidates[0]

def prepare_dataset():
    split = load_split()

    # Reset folders
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)

    # Train: real + CycleGAN
    for name in split["train"]:
        img_file = find_image_by_name(name)
        lbl_file = find_label_by_name(name)

        if img_file is None:
            print(f"[WARN] Missing real image for {name}")
        if lbl_file is None:
            print(f"[WARN] Missing label for {name}")

        # If either missing, skip both real and synthetic to maintain consistency
        if img_file is None or lbl_file is None:
            print(f"[SKIP] {name} due to missing image/label")
            continue

        # Real
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/train" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT))

        # CycleGAN synthetic
        syn_file = find_syn_by_name(name)
        if syn_file:
            cyc_img_name = f"cyc_{syn_file.name}"
            cyc_lbl_name = f"{Path(cyc_img_name).stem}{LBL_EXT}"
            shutil.copy2(syn_file, YOLO_DATA_DIR / "images/train" / cyc_img_name)
            shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_lbl_name)
            print(f"[ADD] CycleGAN {syn_file.name} mapped from {name}")
        else:
            print(f"[MISS] No CycleGAN match for {name}")

    # Val: real only
    for name in split["val"]:
        img_file = find_image_by_name(name)
        lbl_file = find_label_by_name(name)
        if img_file is None or lbl_file is None:
            print(f"[SKIP] val {name} due to missing image/label")
            continue
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/val" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT))

    # Consistency check
    ti = len(list((YOLO_DATA_DIR / "images/train").glob(f"*{IMG_EXT}")))
    tl = len(list((YOLO_DATA_DIR / "labels/train").glob(f"*{LBL_EXT}")))
    vi = len(list((YOLO_DATA_DIR / "images/val").glob(f"*{IMG_EXT}")))
    vl = len(list((YOLO_DATA_DIR / "labels/val").glob(f"*{LBL_EXT}")))
    print(f"Prepared: train {ti} imgs/{tl} labels; val {vi} imgs/{vl} labels")

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
