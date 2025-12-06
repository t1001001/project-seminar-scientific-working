from preprocessing.extract_slices import extract_all
from preprocessing.create_labels import create_all_labels
from utils.cleanup import cleanup

def main():
    extract_all()
    create_all_labels()
    cleanup()

if __name__ == "__main__":
    main()