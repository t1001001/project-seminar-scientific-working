import json
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO
import utils.config as conf
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

PRE_IMG_DIR = Path(f"{conf.ROOT}/data/preprocessed/images")
PRE_LABEL_DIR = Path(f"{conf.ROOT}/data/preprocessed/labels")
SYN_DIR = Path(f"{conf.ROOT}/data/cyclegan/generated")
YOLO_DATA_DIR = Path(f"{conf.ROOT}/data/yolo/cyclegan_aug")
SPLIT_PATH = Path(f"{conf.ROOT}/data/split.json")
YAML_PATH = Path(f"{conf.ROOT}/cyclegan_aug.yaml")

WEIGHTS = "yolo11n.pt"
EPOCHS = 100
IMG_SIZE = 512
BATCH = 32
IMG_EXT = ".png"
LBL_EXT = ".txt"
AUG_MULT = 5
SUFFIXES = ["_fake_A", "_fake_B"]

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def aug_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))

def aug_brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

def aug_gaussian_blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

def aug_gaussian_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def aug_clahe(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    clahe = cv2.createCLAHE(
        clipLimit=random.uniform(2.0, 4.0),
        tileGridSize=(random.choice([8, 16]), random.choice([8, 16]))
    )
    enhanced = clahe.apply(arr)
    return Image.fromarray(enhanced)

AUG_FUNCS = [aug_contrast, aug_brightness, aug_gaussian_blur, aug_gaussian_noise, aug_clahe]

def apply_specific_aug(img_path: Path, out_path: Path, aug_index: int):
    img = Image.open(img_path).convert("L")
    func = AUG_FUNCS[aug_index % len(AUG_FUNCS)]
    img = func(img)
    img.save(out_path)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def prepare_dataset():
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    PRE_IMG_CACHE = {p.name: p for p in PRE_IMG_DIR.rglob(f"*{IMG_EXT}")}
    SYN_CACHE = [p for p in SYN_DIR.rglob(f"*{IMG_EXT}")]
    add_count = skip_count = 0
    ti = tl = 0
    train_names = [n for n in split["train"] if n in PRE_IMG_CACHE]
    print(f"[INFO] Train usable entries: {len(train_names)}")
    for name in train_names:
        if "_" not in name:
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] Invalid image name {name}")
            continue
        uid, name_idx = name.rsplit("_", 1)
        img_file = PRE_IMG_CACHE[name]
        lbl_file = PRE_LABEL_DIR / uid / f"{uid}_{name_idx.replace(IMG_EXT, LBL_EXT)}"
        if not lbl_file.exists():
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] No label found for {name}")
            continue
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/train" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT))
        ti += 1
        tl += 1
        def aug_task(i):
            aug_name = name.replace(IMG_EXT, f"_aug{i+1}{IMG_EXT}")
            apply_specific_aug(img_file, YOLO_DATA_DIR / "images/train" / aug_name, i)
            shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / aug_name.replace(IMG_EXT, LBL_EXT))
        with ThreadPoolExecutor(max_workers=8) as ex:
            ex.map(aug_task, range(AUG_MULT))
        ti += AUG_MULT
        tl += AUG_MULT
    vi = vl = 0
    val_names = [n for n in split["val"] if n in PRE_IMG_CACHE]
    print(f"[INFO] Val usable entries: {len(val_names)}")
    for name in val_names:
        if "_" not in name:
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] Invalid image name {name}")
            continue
        uid, name_idx = name.rsplit("_", 1)
        img_file = PRE_IMG_CACHE[name]
        lbl_file = PRE_LABEL_DIR / uid / f"{uid}_{name_idx.replace(IMG_EXT, LBL_EXT)}"
        if not lbl_file.exists():
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] No label found for {name}")
            continue
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/val" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT))
        vi += 1
        vl += 1
    cyc_imgs = cyc_lbls = 0
    for syn_file in SYN_CACHE:
        stem_base = syn_file.stem
        suf = ""
        for sfx in SUFFIXES:
            if stem_base.endswith(sfx):
                suf = sfx
                stem_base = stem_base[:-len(sfx)]
                break
        if "_" not in stem_base:
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] Invalid synthetic name {syn_file.name}")
            continue
        uid, name_idx = stem_base.rsplit("_", 1)
        lbl_file = PRE_LABEL_DIR / uid / f"{uid}_{name_idx}{LBL_EXT}"
        if not lbl_file.exists():
            skip_count += 1
            if skip_count <= 10:
                print(f"[SKIP] No label found for synthetic {syn_file.name}")
            continue
        cyc_img_name = f"cyc_{stem_base}{suf}{IMG_EXT}"
        cyc_lbl_name = f"cyc_{stem_base}{suf}{LBL_EXT}"
        shutil.copy2(syn_file, YOLO_DATA_DIR / "images/train" / cyc_img_name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_lbl_name)
        add_count += 1
        cyc_imgs += 1
        cyc_lbls += 1
    ti = len(list((YOLO_DATA_DIR / "images/train").glob(f"*{IMG_EXT}")))
    tl = len(list((YOLO_DATA_DIR / "labels/train").glob(f"*{LBL_EXT}")))
    vi = len(list((YOLO_DATA_DIR / "images/val").glob(f"*{IMG_EXT}")))
    vl = len(list((YOLO_DATA_DIR / "labels/val").glob(f"*{LBL_EXT}")))
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
    print("YAML created.")

def yolo_cyclegan_aug():
    prepare_dataset()
    create_yaml()
    model = YOLO(WEIGHTS)
    model.train(data=str(YAML_PATH), epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, name="cyclegan_aug")

if __name__ == "__main__":
    yolo_cyclegan_aug()