import os
import torch
from pathlib import Path
from ultralytics import YOLO
from torchvision.ops import box_iou
import utils.config as conf
import csv

MODELS = {
    "Baseline": f"{conf.ROOT.parent}/runs/detect/baseline/weights/best.pt",
    "Baseline Augmented": f"{conf.ROOT.parent}/runs/detect/baseline_aug/weights/best.pt",
    "Cyclegan": f"{conf.ROOT.parent}/runs/detect/cyclegan/weights/best.pt",
    "Cyclegan Augmented": f"{conf.ROOT.parent}/runs/detect/cyclegan_aug/weights/best.pt",
}

DATA_YAML = f"{conf.ROOT}/luna.yaml"

VAL_IMAGES = f"{conf.ROOT}/data/yolo/baseline/images/val"
VAL_LABELS = f"{conf.ROOT}/data/yolo/baseline/labels/val"

CONF_THRES = 0.001
LOG_CSV = Path(conf.ROOT) / "evaluation_results.csv"

def load_gt_boxes(label_path, img_shape):
    """Load YOLO-format labels and convert to [x1, y1, x2, y2] format"""
    h, w = img_shape
    boxes = []
    if not os.path.exists(label_path):
        return torch.empty((0, 4))
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
    return torch.tensor(boxes, dtype=torch.float32)

def mean_iou_yolo(model_path, val_images, val_labels, conf=0.001):
    """Compute mean IoU ≥ 0.5 between predictions and ground truth"""
    model = YOLO(model_path)
    results = model.predict(
        source=val_images,
        conf=conf,
        stream=True,
        verbose=False
    )
    ious = []
    for r in results:
        if r.boxes is None or r.boxes.xyxy.numel() == 0:
            continue
        preds = r.boxes.xyxy
        h, w = r.orig_shape
        label_file = os.path.join(
            val_labels,
            os.path.basename(r.path).replace(".png", ".txt")
        )
        gts = load_gt_boxes(label_file, (h, w)).to(preds.device)
        if gts.numel() == 0:
            continue
        iou_matrix = box_iou(preds, gts)
        best_iou = iou_matrix.max(dim=1)[0]
        best_iou = best_iou[best_iou >= 0.5]
        ious.append(best_iou)
    return torch.cat(ious).mean().item() if ious else 0.0

def evaluate(model_name, model_path):
    print(f"\nEvaluating {model_name}...")
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, verbose=False)
    ap50 = float(metrics.box.map50)
    ap5095 = float(metrics.box.map)
    miou = mean_iou_yolo(model_path, VAL_IMAGES, VAL_LABELS, conf=CONF_THRES)
    print(f"{model_name}: AP50={ap50:.3f}, AP50-95={ap5095:.3f}, mIoU={miou:.3f}")
    return ap50, ap5095, miou

def log_results(results, csv_path=LOG_CSV):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "AP50", "AP50-95", "mIoU"])
        for name, (ap50, ap5095, miou) in results.items():
            writer.writerow([name, f"{ap50:.3f}", f"{ap5095:.3f}", f"{miou:.3f}"])
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    print("\n=== EVALUATION OF ALL EXPERIMENTS ===")
    results = {}
    for name, path in MODELS.items():
        results[name] = evaluate(name, path)
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Model':35s} AP@0.5   AP@0.5:0.95   Mean IoU")
    for name, (ap50, ap5095, miou) in results.items():
        print(f"{name:35s} {ap50:.3f}     {ap5095:.3f}         {miou:.3f}")
    log_results(results)