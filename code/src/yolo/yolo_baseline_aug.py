import json
import shutil
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

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def aug_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(np.random.uniform(0.5, 1.5))

def aug_brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(np.random.uniform(0.5, 1.5))

def aug_gaussian_blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5)))

def aug_gaussian_noise(img: Image.Image) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    sigma = np.random.uniform(5, 15)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def aug_clahe(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    clahe = cv2.createCLAHE(
        clipLimit=np.random.uniform(2.0, 4.0),
        tileGridSize=(np.random.choice([8, 16]), np.random.choice([8, 16]))
    )
    return Image.fromarray(clahe.apply(arr))

AUG_FUNCS = [aug_contrast, aug_brightness, aug_gaussian_blur, aug_gaussian_noise, aug_clahe]

def apply_specific_aug(img_path: Path, out_path: Path, aug_index: int):
    img = Image.open(img_path).convert("L")
    func = AUG_FUNCS[aug_index % len(AUG_FUNCS)]
    func(img).save(out_path)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def prepare_dataset():
    split = load_split()
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    IMG_CACHE = {p.name: p for p in PRE_IMG_DIR.glob("*/*")}
    LBL_CACHE = {p.name: p for p in PRE_LABEL_DIR.glob("*/*")}
    train_pairs = []
    aug_tasks = []
    for name in split["train"]:
        img_file = IMG_CACHE.get(name)
        lbl_file = LBL_CACHE.get(name.replace(IMG_EXT, LBL_EXT))
        if img_file is None or lbl_file is None:
            continue
        train_pairs.append((img_file, YOLO_DATA_DIR / "images/train" / name))
        train_pairs.append((lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT)))
        for i in range(AUG_MULT):
            aug_name = name.replace(IMG_EXT, f"_aug{i+1}{IMG_EXT}")
            aug_tasks.append((img_file, YOLO_DATA_DIR / "images/train" / aug_name, i, lbl_file))
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(lambda p: shutil.copy2(p[0], p[1]), train_pairs)
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(lambda p: (apply_specific_aug(p[0], p[1], p[2]), shutil.copy2(p[3], p[1].with_suffix(LBL_EXT))), aug_tasks)
    val_pairs = []
    for name in split["val"]:
        img_file = IMG_CACHE.get(name)
        lbl_file = LBL_CACHE.get(name.replace(IMG_EXT, LBL_EXT))
        if img_file is None or lbl_file is None:
            continue
        val_pairs.append((img_file, YOLO_DATA_DIR / "images/val" / name))
        val_pairs.append((lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT)))
    with ThreadPoolExecutor(max_workers=8) as ex:
        ex.map(lambda p: shutil.copy2(*p), val_pairs)
    train_imgs = list((YOLO_DATA_DIR / "images/train").glob(f"*{IMG_EXT}"))
    train_lbls = list((YOLO_DATA_DIR / "labels/train").glob(f"*{LBL_EXT}"))
    val_imgs = list((YOLO_DATA_DIR / "images/val").glob(f"*{IMG_EXT}"))
    val_lbls = list((YOLO_DATA_DIR / "labels/val").glob(f"*{LBL_EXT}"))
    print(f"Train: {len(train_imgs)} images, {len(train_lbls)} labels; Val: {len(val_imgs)} images, {len(val_lbls)} labels")

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

def yolo_baseline_aug():
    prepare_dataset()
    create_yaml()
    model = YOLO(WEIGHTS)
    model.train(data=str(YAML_PATH), epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, name="baseline_aug")

if __name__ == "__main__":
    yolo_baseline_aug()
