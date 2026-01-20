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
YOLO_DATA_DIR = Path(f"{conf.ROOT}/data/yolo/baseline_aug")
SPLIT_PATH = Path(f"{conf.ROOT}/data/split.json")
YAML_PATH = Path(f"{conf.ROOT}/baseline_aug.yaml")

WEIGHTS = "yolo11n.pt"
EPOCHS = 100
IMG_SIZE = 512
BATCH = 32
IMG_EXT = ".png"
LBL_EXT = ".txt"
AUG_MULT = 5
MAX_WORKERS = 8

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def aug_contrast(img):
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.5, 1.5))

def aug_brightness(img):
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

def aug_gaussian_blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

def aug_gaussian_noise(img):
    arr = np.array(img, dtype=np.float32)
    sigma = random.uniform(5, 15)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def aug_clahe(img):
    arr = np.array(img)
    clahe = cv2.createCLAHE(
        clipLimit=random.uniform(2.0, 4.0),
        tileGridSize=(random.choice([8, 16]), random.choice([8, 16]))
    )
    return Image.fromarray(clahe.apply(arr))

AUG_FUNCS = [
    aug_contrast,
    aug_brightness,
    aug_gaussian_blur,
    aug_gaussian_noise,
    aug_clahe,
]

def apply_specific_aug(img_path: Path, out_path: Path, aug_index: int):
    img = Image.open(img_path).convert("L")
    func = AUG_FUNCS[aug_index % len(AUG_FUNCS)]
    func(img).save(out_path)

def prepare_dataset():
    split = load_split()
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        ensure_dir(YOLO_DATA_DIR / folder)
    IMG_CACHE = {p.name: p for p in PRE_IMG_DIR.glob("*/*")}
    LBL_CACHE = {p.name: p for p in PRE_LABEL_DIR.glob("*/*")}
    copy_tasks = []
    aug_tasks = []
    for name in split["train"]:
        img_file = IMG_CACHE.get(name)
        lbl_file = LBL_CACHE.get(name.replace(IMG_EXT, LBL_EXT))
        if not img_file or not lbl_file:
            continue
        copy_tasks.append((img_file, YOLO_DATA_DIR / "images/train" / name))
        copy_tasks.append((lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT)))
        for i in range(AUG_MULT):
            aug_name = name.replace(IMG_EXT, f"_aug{i+1}{IMG_EXT}")
            aug_img_path = YOLO_DATA_DIR / "images/train" / aug_name
            aug_lbl_path = YOLO_DATA_DIR / "labels/train" / aug_name.replace(IMG_EXT, LBL_EXT)
            aug_tasks.append((img_file, aug_img_path, i, lbl_file, aug_lbl_path))
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        ex.map(lambda p: shutil.copy2(p[0], p[1]), copy_tasks)
    def aug_worker(task):
        img_file, out_img, idx, lbl_file, out_lbl = task
        apply_specific_aug(img_file, out_img, idx)
        shutil.copy2(lbl_file, out_lbl)
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        ex.map(aug_worker, aug_tasks)
    val_tasks = []
    for name in split["val"]:
        img_file = IMG_CACHE.get(name)
        lbl_file = LBL_CACHE.get(name.replace(IMG_EXT, LBL_EXT))
        if not img_file or not lbl_file:
            continue
        val_tasks.append((img_file, YOLO_DATA_DIR / "images/val" / name))
        val_tasks.append((lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT)))
    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        ex.map(lambda p: shutil.copy2(p[0], p[1]), val_tasks)
    print(
        f"Train: {len(list((YOLO_DATA_DIR / 'images/train').glob('*')))} images, "
        f"{len(list((YOLO_DATA_DIR / 'labels/train').glob('*')))} labels; "
        f"Val: {len(list((YOLO_DATA_DIR / 'images/val').glob('*')))} images, "
        f"{len(list((YOLO_DATA_DIR / 'labels/val').glob('*')))} labels"
    )

def create_yaml():
    ensure_dir(YAML_PATH.parent)
    YAML_PATH.write_text(
        f"""train: {YOLO_DATA_DIR}/images/train
val: {YOLO_DATA_DIR}/images/val
nc: 1
names: ["nodule"]"""
    )

def yolo_baseline_aug():
    prepare_dataset()
    create_yaml()
    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="baseline_aug"
    )

if __name__ == "__main__":
    yolo_baseline_aug()
