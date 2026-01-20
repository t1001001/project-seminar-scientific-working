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
    split = load_split()
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)
    PRE_IMG_CACHE = {p.name: p for p in PRE_IMG_DIR.rglob(f"*{IMG_EXT}")}
    PRE_LABEL_CACHE = {p.name: p for p in PRE_LABEL_DIR.rglob(f"*{LBL_EXT}")}
    SYN_CACHE = [p for p in SYN_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    ti = tl = vi = vl = cyc_imgs = cyc_lbls = 0
    add_count = miss_count = skip_count = 0
    train_names = [n for n in split["train"] if n in PRE_IMG_CACHE and n.replace(IMG_EXT, LBL_EXT) in PRE_LABEL_CACHE]
    for name in train_names:
        img_file = PRE_IMG_CACHE[name]
        lbl_file = PRE_LABEL_CACHE[name.replace(IMG_EXT, LBL_EXT)]
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
        stem = Path(name).stem
        syn_files = [p for p in SYN_CACHE if p.stem.startswith(stem + "_") and any(p.stem.endswith(s) for s in SUFFIXES)]
        if syn_files:
            pairs = []
            for syn_file in syn_files:
                s = syn_file.stem
                suf = next((sx for sx in SUFFIXES if s.endswith(sx)), "")
                cyc_img_name = f"cyc_{stem}{suf}{syn_file.suffix}"
                cyc_lbl_name = f"cyc_{stem}{suf}{LBL_EXT}"
                pairs.append((syn_file, YOLO_DATA_DIR / "images/train" / cyc_img_name))
                pairs.append((lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_lbl_name))
                cyc_imgs += 1
                cyc_lbls += 1
            with ThreadPoolExecutor(max_workers=8) as ex:
                ex.map(lambda p: shutil.copy2(*p), pairs)
            add_count += len(syn_files)
        else:
            miss_count += 1
    val_names = [n for n in split["val"] if n in PRE_IMG_CACHE and n.replace(IMG_EXT, LBL_EXT) in PRE_LABEL_CACHE]
    for name in val_names:
        img_file = PRE_IMG_CACHE[name]
        lbl_file = PRE_LABEL_CACHE[name.replace(IMG_EXT, LBL_EXT)]
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/val" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/val" / name.replace(IMG_EXT, LBL_EXT))
        vi += 1
        vl += 1
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

def yolo_cyclegan_aug():
    prepare_dataset()
    create_yaml()
    model = YOLO(WEIGHTS)
    model.train(data=str(YAML_PATH), epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH, name="cyclegan_aug")

if __name__ == "__main__":
    yolo_cyclegan_aug()