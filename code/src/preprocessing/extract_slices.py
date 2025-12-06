import os
import SimpleITK as sitk
import numpy as np
import cv2
from tqdm import tqdm
import src.utils.config as conf

RAW_DIR = f"{conf.ROOT}/data/raw/luna16"
OUT_DIR = f"{conf.ROOT}/data/preprocessed"
OUTPUT_IMG_DIR = os.path.join(OUT_DIR, "images")

HU_MIN = -1000
HU_MAX = 400

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize_hu(arr):
    arr = np.clip(arr, HU_MIN, HU_MAX)
    arr = (arr - HU_MIN) / (HU_MAX - HU_MIN)
    arr = (arr * 255).astype(np.uint8)
    return arr

def load_mhd(mhd_path):
    image = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(image)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    return arr, origin, spacing

def extract_all():
    print("Extracting all LUNA16 CT slices...")

    ensure_dir(OUTPUT_IMG_DIR)

    for subset_id in range(10):
        subset_dir = os.path.join(RAW_DIR, f"subset{subset_id}")
        print(f"\n Processing {subset_dir}")

        scans = []
        for root, dirs, files in os.walk(subset_dir):
            for f in files:
                if f.endswith(".mhd"):
                    scans.append(os.path.join(root, f))

        if not scans:
            print(f"[WARNING] No .mhd scans found in {subset_dir}")
            continue

        for scan_path in tqdm(scans, desc=f"subset{subset_id}"):
            scan_file = os.path.basename(scan_path)
            scan_id = scan_file.replace(".mhd", "")

            arr, origin, spacing = load_mhd(scan_path)
            arr = normalize_hu(arr)

            scan_out_dir = os.path.join(OUTPUT_IMG_DIR, scan_id)
            ensure_dir(scan_out_dir)

            for i, slice_img in enumerate(arr):
                slice_filename = f"{scan_id}_{i:04d}.png"
                slice_path = os.path.join(scan_out_dir, slice_filename)
                cv2.imwrite(slice_path, slice_img)

    print("\n DONE! All slices extracted")
    print(f"Saved under: {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    extract_all()