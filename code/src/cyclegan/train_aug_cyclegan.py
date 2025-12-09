from ultralytics import YOLO
from pathlib import Path
import utils.config as conf

YAML_PATH = Path(f"{conf.ROOT}/luna_aug.yaml")
YOLO_DATA_DIR = Path(f"{conf.ROOT}/data/yolo/cyclegan_aug")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def create_yaml():
    print(f"Creating dataset YAML: {YAML_PATH}")
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

def train_aug_cyclegan():
    create_yaml()
    print("Training YOLOv11 on CycleGAN-Augmented Dataset")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(YAML_PATH),
        epochs=50,
        imgsz=512,
        batch=32,
        name="luna11_cyclegan_aug"
    )

if __name__ == "__main__":
    train_aug_cyclegan()