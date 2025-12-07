from preprocessing.extract_slices import extract_all
from preprocessing.create_labels import create_all_labels
from utils.cleanup import cleanup
from cyclegan.prepare_cyclegan import prepare_cyclegan
from cyclegan.train_cyclegan import train_cyclegan
from cyclegan.train_aug_cyclegan import train_aug_cyclegan
from augmentation.augment_cyclegan import augment_cyclegan
from augmentation.augment_yolo import augment_yolo

def main():
    extract_all()
    create_all_labels()
    cleanup()
    prepare_cyclegan()
    train_cyclegan()
    augment_cyclegan()
    augment_yolo()
    train_aug_cyclegan()

if __name__ == "__main__":
    main()