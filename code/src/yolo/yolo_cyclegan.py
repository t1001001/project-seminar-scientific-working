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
SUFFIXES = ["_fake_A", "_fake_B"]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def prepare_dataset():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    for folder in ["images/train","images/val","labels/train","labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR/folder,ignore_errors=True)
        (YOLO_DATA_DIR/folder).mkdir(parents=True,exist_ok=True)
    img_index = {p.name:p for p in PRE_IMG_DIR.rglob(f"*{IMG_EXT}")}
    add_count = 0
    skip_count = 0
    train_names = [n for n in split["train"] if n in img_index]
    print(f"[INFO] Train usable entries: {len(train_names)}")
    for name in train_names:
        img_file = img_index[name]
        if "_" not in name:
            skip_count += 1
            if skip_count <= 10: print(f"[SKIP] Invalid image name {name}")
            continue
        uid, name_idx = name.rsplit("_",1)
        lbl_file = PRE_LABEL_DIR/uid/f"{uid}_{name_idx.replace(IMG_EXT,LBL_EXT)}"
        if not lbl_file.exists():
            skip_count += 1
            if skip_count <= 10: print(f"[SKIP] No label found for {name}")
            continue
        shutil.copy2(img_file, YOLO_DATA_DIR/"images/train"/name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR/"labels/train"/name.replace(IMG_EXT,LBL_EXT))
    val_names = [n for n in split["val"] if n in img_index]
    print(f"[INFO] Val usable entries: {len(val_names)}")
    for name in val_names:
        img_file = img_index[name]
        if "_" not in name:
            skip_count += 1
            if skip_count <= 10: print(f"[SKIP] Invalid image name {name}")
            continue
        uid, name_idx = name.rsplit("_",1)
        lbl_file = PRE_LABEL_DIR/uid/f"{uid}_{name_idx.replace(IMG_EXT,LBL_EXT)}"
        if not lbl_file.exists():
            skip_count += 1
            if skip_count <= 10: print(f"[SKIP] No label found for {name}")
            continue
        shutil.copy2(img_file, YOLO_DATA_DIR/"images/val"/name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR/"labels/val"/name.replace(IMG_EXT,LBL_EXT))
    syn_files = list(SYN_DIR.glob(f"*{IMG_EXT}"))
    for syn_file in syn_files:
        stem_base = syn_file.stem
        suf = ""
        for sfx in SUFFIXES:
            if stem_base.endswith(sfx):
                suf = sfx
                stem_base = stem_base[:-len(sfx)]
                break
        if "_" not in stem_base:
            skip_count += 1
            if skip_count <= 10: print(f"[SKIP] Invalid synthetic name {syn_file.name}")
            continue
        uid, name_idx = stem_base.rsplit("_",1)
        lbl_file = PRE_LABEL_DIR/uid/f"{uid}_{name_idx}{LBL_EXT}"
        if not lbl_file.exists():
            skip_count += 1
            if skip_count <= 10: print(f"[SKIP] No label found for synthetic {syn_file.name}")
            continue
        cyc_img_name = f"cyc_{stem_base}{suf}{IMG_EXT}"
        cyc_lbl_name = f"cyc_{stem_base}{suf}{LBL_EXT}"
        shutil.copy2(syn_file, YOLO_DATA_DIR/"images/train"/cyc_img_name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR/"labels/train"/cyc_lbl_name)
        add_count += 1
    ti = len(list((YOLO_DATA_DIR/"images/train").glob(f"*{IMG_EXT}")))
    tl = len(list((YOLO_DATA_DIR/"labels/train").glob(f"*{LBL_EXT}")))
    vi = len(list((YOLO_DATA_DIR/"images/val").glob(f"*{IMG_EXT}")))
    vl = len(list((YOLO_DATA_DIR/"labels/val").glob(f"*{LBL_EXT}")))
    cyc_imgs = len(list((YOLO_DATA_DIR/"images/train").glob("cyc_*")))
    cyc_lbls = len(list((YOLO_DATA_DIR/"labels/train").glob("cyc_*")))
    print(f"Prepared: train {ti} imgs/{tl} labels; val {vi} imgs/{vl} labels")
    print(f"Synthetic added: {add_count} | skips (no label or invalid): {skip_count}")
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

def yolo_cyclegan():
    prepare_dataset()
    create_yaml()
    model = YOLO(WEIGHTS)
    model.train(data=str(YAML_PATH), epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, name="cyclegan")

if __name__ == "__main__":
    yolo_cyclegan()