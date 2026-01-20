from preprocessing.extract_slices import extract_all
from preprocessing.create_labels import create_all_labels
from utils.cleanup import cleanup
from utils.make_split import make_split
from cyclegan.cyclegan_prep import cyclegan_prep
from cyclegan.cyclegan_train import cyclegan_train
from cyclegan.cyclegan_generate import cyclegan_generate
from yolo.yolo_baseline import yolo_baseline              # Experiment A
from yolo.yolo_baseline_aug import yolo_baseline_aug      # Experiment B
from yolo.yolo_cyclegan import yolo_cyclegan            # Experiment C
from yolo.yolo_cyclegan_aug import yolo_cyclegan_aug    # Experiment D

from evaluation.evaluation import evaluate
import utils.config as conf

MODELS = {
    "Baseline": f"{conf.ROOT.parent}/runs/detect/baseline/weights/best.pt",
    "Baseline Augmented": f"{conf.ROOT.parent}/runs/detect/baseline_aug/weights/best.pt",
    "Cyclegan": f"{conf.ROOT.parent}/runs/detect/cyclegan/weights/best.pt",
    "Cyclegan Augmented": f"{conf.ROOT.parent}/runs/detect/cyclegan_aug/weights/best.pt",
}

def main():
    extract_all()
    create_all_labels()
    make_split()
    cleanup()
    cyclegan_prep()
    cyclegan_train()
    cyclegan_generate()
    yolo_baseline()
    yolo_baseline_aug()
    yolo_cyclegan()
    yolo_cyclegan_aug()
    for name, model_path in MODELS.items():
        evaluate(name, model_path)

if __name__ == "__main__":
    main()