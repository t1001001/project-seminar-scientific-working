from preprocessing.extract_slices import extract_all
from preprocessing.create_labels import create_all_labels
from utils.make_split import make_split 
from utils.cleanup import cleanup

def main():
    extract_all()
    create_all_labels()
    make_split()
    cleanup()

if __name__ == "__main__":
    main()