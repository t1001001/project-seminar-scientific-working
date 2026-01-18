import json
import shutil
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO
import utils.config as conf
import numpy as np
import cv2

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
AUG_MULT = 5          # number of augmented copies per real train image
AUGMENT_CYC = False   # set True to also augment CycleGAN images

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
    clahe = cv2.createCLAHE(clipLimit=random.uniform(2.0, 4.0),
                            tileGridSize=(random.choice([8, 16]), random.choice([8, 16])))
    enhanced = clahe.apply(arr)
    return Image.fromarray(enhanced)

AUG_FUNCS = [
    aug_contrast,         # 1
    aug_brightness,       # 2
    aug_gaussian_blur,    # 3
    aug_gaussian_noise,   # 4
    aug_clahe,            # 5
]
def apply_specific_aug(img_path: Path, out_path: Path, aug_index: int):
    img = Image.open(img_path).convert("L")
    func = AUG_FUNCS[aug_index % len(AUG_FUNCS)]
    img = func(img)
    img.save(out_path)

def load_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)

def find_image_by_name(name: str) -> Path | None:
    m = list(PRE_IMG_DIR.glob(f"*/{name}"))
    return m[0] if m else None

def find_label_by_name(name: str) -> Path | None:
    m = list(PRE_LABEL_DIR.glob(f"*/{name.replace(IMG_EXT, LBL_EXT)}"))
    return m[0] if m else None

def prepare_dataset():
    split = load_split()
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        shutil.rmtree(YOLO_DATA_DIR / folder, ignore_errors=True)
        (YOLO_DATA_DIR / folder).mkdir(parents=True, exist_ok=True)

    # Train: real + specific augmented copies + CycleGAN
    for name in split["train"]:
        img_file = find_image_by_name(name)
        lbl_file = find_label_by_name(name)
        if img_file is None or lbl_file is None:
            continue

        # Real
        shutil.copy2(img_file, YOLO_DATA_DIR / "images/train" / name)
        shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / name.replace(IMG_EXT, LBL_EXT))

        # Augmented copies from real (contrast, brightness, blur, cycling if AUG_MULT>3)
        for i in range(AUG_MULT):
            aug_name = name.replace(IMG_EXT, f"_aug{i+1}{IMG_EXT}")
            apply_specific_aug(img_file, YOLO_DATA_DIR / "images/train" / aug_name, i)
            shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / aug_name.replace(IMG_EXT, LBL_EXT))

        # CycleGAN synthetic (prefixed)
        syn_file = SYN_DIR / name
        if syn_file.exists():
            cyc_name = f"cyc_{name}"
            shutil.copy2(syn_file, YOLO_DATA_DIR / "images/train" / cyc_name)
            shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_name.replace(IMG_EXT, LBL_EXT))

            # Optional: also augment synthetic images (same specific types)
            if AUGMENT_CYC:
                for i in range(AUG_MULT):
                    cyc_aug = name.replace(IMG_EXT, f"_cyc_aug{i+1}{IMG_EXT}")
                    apply_specific_aug(syn_file, YOLO_DATA_DIR / "images/train" / cyc_aug, i)
                    shutil.copy2(lbl_file, YOLO_DATA_DIR / "labels/train" / cyc_aug.replace(IMG_EXT, LBL_EXT))

    # Val: real only
    for name in split["val"]:
        img_file = find_image_by_name(name)
        lbl_file = find_label_by_name(name)
        if img_file is None or lbl_file is None:
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

def yolo_cyclegan_aug():
    print("\n=== YOLO Baseline Augmentation + CycleGAN (Experiment D) ===")
    prepare_dataset()
    create_yaml()

    model = YOLO(WEIGHTS)
    model.train(
        data=str(YAML_PATH),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        name="cyclegan_aug"
    )
    print("=== EXPERIMENT D DONE ===\n")

if __name__ == "__main__":
    yolo_cyclegan_aug()
