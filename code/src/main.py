from preprocessing.extract_slices import extract_all
from preprocessing.create_labels import create_all_labels
from utils.cleanup import cleanup
from cyclegan.prepare_cyclegan import prepare_cyclegan
from cyclegan.train_cyclegan import train_cyclegan
from augmentation.augment_cyclegan import augment_cyclegan
from augmentation.augment_yolo import augment_yolo
from yolo.yolo_baseline import yolo_baseline
from yolo.yolo_aug import yolo_aug
from evaluation.evaluation import evaluate

def main():
    extract_all()
    create_all_labels()
    cleanup()
    yolo_baseline()
    prepare_cyclegan()
    train_cyclegan()
    augment_cyclegan()
    augment_yolo()
    yolo_aug()
    evaluate()

if __name__ == "__main__":
    main()