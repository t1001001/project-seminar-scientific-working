import numpy as np
from ultralytics import YOLO
import utils.config as conf


MODELS = {
    "Baseline": f"{conf.ROOT}/runs/detect/luna11_baseline/weights/best.pt",
    "Augmented": f"{conf.ROOT}/runs/detect/luna11_cyclegan_aug/weights/best.pt",
}

DATA_YAML = "yolo_training/configs/luna.yaml"
IOU_MATCH_THRESHOLD = 0.5

def compute_iou(b1, b2):
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def evaluate(model_name, model_path):
    print(f"\n=== Evaluating {model_name} ===")
    model = YOLO(model_path)
    metrics = model.val(
        data=DATA_YAML,
        save_json=True,
        verbose=False
    )
    ap50 = metrics.box.map50
    ap5095 = metrics.box.map
    ious = []
    for pred in metrics.pred:
        if pred is None or pred.boxes is None:
            continue
        gt_boxes = pred.gt_boxes.xyxy.cpu().numpy()
        pr_boxes = pred.boxes.xyxy.cpu().numpy()
        for pb in pr_boxes:
            best_iou = 0
            for gt in gt_boxes:
                best_iou = max(best_iou, compute_iou(pb, gt))
            if best_iou >= IOU_MATCH_THRESHOLD:
                ious.append(best_iou)
    mean_iou = np.mean(ious) if ious else 0
    return ap50, ap5095, mean_iou

if __name__ == "__main__":
    print("\nFINAL EVALUATION (AP + IoU)\n")
    results = {}
    for name, path in MODELS.items():
        results[name] = evaluate(name, path)
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Model':35s} AP@0.5   AP@0.5:0.95   Mean IoU")
    for name, (ap50, ap5095, miou) in results.items():
        print(f"{name:35s} {ap50:.3f}     {ap5095:.3f}         {miou:.3f}")