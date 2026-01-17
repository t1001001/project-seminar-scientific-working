import os
from pathlib import Path

IMG_DIR = Path("data/preprocessed/images")
OUT_LABEL_DIR = Path("data/preprocessed/labels")

def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)

def fill_missing_empty_labels():
    for scan_dir in IMG_DIR.iterdir():
        if not scan_dir.is_dir(): continue
        lbl_dir = OUT_LABEL_DIR / scan_dir.name
        ensure_dir(lbl_dir)
        for img_path in scan_dir.glob("*.png"):
            lbl_path = lbl_dir / img_path.name.replace(".png", ".txt")
            if not lbl_path.exists():
                open(lbl_path, "w").close()  # create empty label (negative slice)

if __name__ == "__main__":
    fill_missing_empty_labels()
