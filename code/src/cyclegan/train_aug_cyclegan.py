from ultralytics import YOLO

def train_aug_cyclegan():
    print("Training YOLOv11 on CycleGAN-Augmented Dataset")

    model = YOLO("yolov11n.pt")
    model.train(
        data="yolo_training/configs/luna_aug.yaml",
        epochs=50,
        imgsz=512,
        batch=32,
        name="luna11_cyclegan_aug"
    )

if __name__ == "__main__":
    train_aug_cyclegan()