import json
import shutil
from pathlib import Path
from ultralytics import YOLO
import utils.config as conf
from concurrent.futures import ThreadPoolExecutor

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

def copy_files_parallel(pairs):
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(lambda p: shutil.copy2(*p), pairs)

def prepare_dataset():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    img_index = {p.name: p for p in PRE_IMG_DIR.rglob(f"*{IMG_EXT}")}
    lbl_index = {p.name: p for p in PRE_LABEL_DIR.rglob(f"*{LBL_EXT}")}
    SYN_CACHE = [p for p in SYN_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    add_count = skip_count = 0
    ti = tl = vi = vl = 0
    cyc_imgs = cyc_lbls = 0
    train_names = [n for n in split["train"] if n in img_index and n.replace(IMG_EXT, LBL_EXT) in lbl_index]
    pairs = []
    for name in train_names:
        img_file = img_index[name]
        lbl_file = lbl_index[name.replace(IMG_EXT, LBL_EXT)]
        pairs.append((img_file, YOLO_DATA_DIR / "images/train" / name))
        pairs.append((lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT)))
        ti += 1
        tl += 1
    copy_files_parallel(pairs)
    for name in train_names:
        stem = Path(name).stem
        syn_files = [p for p in SYN_CACHE if p.stem.startswith(stem + "_") and any(p.stem.endswith(s) for s in SUFFIXES)]
        if syn_files:
            pairs = []
            for syn_file in syn_files:
                s = syn_file.stem
                suf = next((sx for sx in SUFFIXES if s.endswith(sx)), "")
                cyc_img_name = f"cyc_{stem}{suf}{syn_file.suffix}"
                cyc_lbl_name = f"cyc_{stem}{suf}{LBL_EXT}"
                lbl_file = lbl_index[name.replace(IMG_EXT, LBL_EXT)]
                pairs.append((syn_file, YOLO_DATA_DIR / "images/train" / cyc_img_name))
                pairs.append((lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_lbl_name))
                cyc_imgs += 1
                cyc_lbls += 1
            copy_files_parallel(pairs)
            add_count += len(syn_files)
    val_names = [n for n in split["val"] if n in img_index and n.replace(IMG_EXT, LBL_EXT) in lbl_index]
    pairs = []
    for name in val_names:
        img_file = img_index[name]
        lbl_file = lbl_index[name.replace(IMG_EXT, LBL_EXT)]
        pairs.append((img_file, YOLO_DATA_DIR / "images/val" / name))
        pairs.append((lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT)))
        vi += 1
        vl += 1
    copy_files_parallel(pairs)
    print(f"Prepared: train {ti} imgs/{tl} labels; val {vi} imgs/{vl} labels")
    print(f"Synthetic added: {add_count} | skips: {skip_count}")
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