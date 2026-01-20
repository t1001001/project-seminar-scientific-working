import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import utils.config as conf
import re

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

SUFFIXES = ["_fake_A", "_fake_B", "_real_A", "_real_B", "_rec_A", "_rec_B"]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def find_image_by_name(name: str) -> Path | None:
    matches = list(PRE_IMG_DIR.rglob(name))
    return matches[0] if matches else None

def find_label_by_name(name: str) -> Path | None:
    matches = list(PRE_LABEL_DIR.rglob(name.replace(IMG_EXT, LBL_EXT)))
    return matches[0] if matches else None

def find_syn_by_names(name: str) -> list[Path]:
    stem = Path(name).stem
    exts = {".png", ".jpg", ".jpeg"}

    candidates = [
        p for p in SYN_DIR.rglob("*")
        if p.suffix.lower() in exts
        and p.stem.startswith(stem + "_")
        and any(p.stem.endswith(suf) for suf in SUFFIXES)
    ]

    return sorted(candidates)

def prepare_dataset():
    split = load_split()
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    add_count = 0
    miss_count = 0
    skip_count = 0
    train_names = [n for n in split["train"] if find_image_by_name(n) and find_label_by_name(n)]
    print(f"[INFO] Train usable entries: {len(train_names)}")
    for name in train_names:
        img_file = find_image_by_name(name)
        lbl_file = find_label_by_name(name)
        if img_file is None or lbl_file is None:
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] {name} due to missing image/label")
            continue
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/train" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT))
        syn_files = find_syn_by_names(name)
        if syn_files:
            orig = Path(name).stem
            for syn_file in syn_files:
                try:
                    s = syn_file.stem
                    suf = next((sx for sx in SUFFIXES if s.endswith(sx)), "")
                    cyc_img_name = f"cyc_{orig}{suf}{syn_file.suffix}"
                    cyc_lbl_name = f"cyc_{orig}{suf}{LBL_EXT}"

                    shutil.copy2(syn_file, YOLO_DATA_DIR / "images/train" / cyc_img_name)
                    shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_lbl_name)

                    print(f"[ADD] CycleGAN {syn_file.name} -> {cyc_img_name}")
                    add_count += 1

                except Exception as e:
                    print(f"[ERR] Failed copying {syn_file}: {e}")
        else:
            miss_count += 1
            if miss_count <= 10:
                print(f"[MISS] No CycleGAN match for {name}")
    val_names = [n for n in split["val"] if find_image_by_name(n) and find_label_by_name(n)]
    print(f"[INFO] Val usable entries: {len(val_names)}")
    for name in val_names:
        img_file = find_image_by_name(name)
        lbl_file = find_label_by_name(name)
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/val" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT))
    ti = len(list((YOLO_DATA_DIR / "images/train").glob(f"*{IMG_EXT}")))
    tl = len(list((YOLO_DATA_DIR / "labels/train").glob(f"*{LBL_EXT}")))
    vi = len(list((YOLO_DATA_DIR / "images/val").glob(f"*{IMG_EXT}")))
    vl = len(list((YOLO_DATA_DIR / "labels/val").glob(f"*{LBL_EXT}")))
    cyc_imgs = len(list((YOLO_DATA_DIR / "images/train").glob("cyc_*")))
    cyc_lbls = len(list((YOLO_DATA_DIR / "labels/train").glob("cyc_*")))
    print(f"Prepared: train {ti} imgs/{tl} labels; val {vi} imgs/{vl} labels")
    print(f"Synthetic added: {add_count} | misses: {miss_count} | skips: {skip_count}")
    print(f"cyc_* images: {cyc_imgs} | cyc_* labels: {cyc_lbls}")

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
    prepare_dataset()
    create_yaml()
    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="cyclegan"
    )

if __name__ == "__main__":
    yolo_cyclegan()