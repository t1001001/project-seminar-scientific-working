import shutil
from pathlib import Path
import utils.config as conf

DATASET_PATH = f"{conf.ROOT}/raw"

def cleanup():
    folder = Path(DATASET_PATH) / "luna16"
    if folder.exists():
        shutil.rmtree(folder)
        print(f"Deleted folder: {folder}")
    else:
        print(f"Folder does not exist: {folder}")

if __name__ == "__main__":
    cleanup()