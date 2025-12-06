import os
import csv
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import utils.config as conf

RAW_DIR = f"{conf.ROOT}/data/raw/luna16"
IMG_DIR = f"{conf.ROOT}/data/preprocessed/images"
OUT_LABEL_DIR = f"{conf.ROOT}/data/preprocessed/labels"

ANNOTATION_FILE = os.path.join(RAW_DIR, "annotations.csv")

HU_MIN = -1000
HU_MAX = 400

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_mhd_info(mhd_path):
    img = sitk.ReadImage(mhd_path)
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection())
    arr = sitk.GetArrayFromImage(img)
    return arr, origin, spacing, direction

def world_to_voxel(world_coord, origin, spacing):
    return (world_coord - origin) / spacing

def voxel_to_world(voxel_coord, origin, spacing):
    return voxel_coord * spacing + origin

def create_all_labels():
    print("Generating YOLO labels from LUNA16 annotations...")
    ensure_dir(OUT_LABEL_DIR)

    print("Loading annotations CSV...")
    annotations = []
    with open(ANNOTATION_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            scan_id = row[0]
            x, y, z = map(float, row[1:4])
            d = float(row[4])
            annotations.append((scan_id, x, y, z, d))

    print("\nProcessing scans...")

    for scan_id in tqdm(os.listdir(IMG_DIR)):
        scan_img_dir = os.path.join(IMG_DIR, scan_id)
        if not os.path.isdir(scan_img_dir):
            continue

        mhd_path = None
        for subset_id in range(10):
            subset_dir = os.path.join(RAW_DIR, f"subset{subset_id}")
            for root, dirs, files in os.walk(subset_dir):
                candidate = os.path.join(root, scan_id + ".mhd")
                if os.path.exists(candidate):
                    mhd_path = candidate
                    break
            if mhd_path:
                break

        if mhd_path is None:
            print(f"[WARNING] Missing MHD for scan {scan_id}, skipping.")
            continue

        volume, origin, spacing, direction = load_mhd_info(mhd_path)
        depth = volume.shape[0]
        height = volume.shape[1]
        width = volume.shape[2]

        label_out_dir = os.path.join(OUT_LABEL_DIR, scan_id)
        ensure_dir(label_out_dir)

        scan_annots = [a for a in annotations if a[0] == scan_id]

        for (scan_id, x_world, y_world, z_world, d_mm) in scan_annots:
            voxel = world_to_voxel(
                np.array([x_world, y_world, z_world]),
                origin,
                spacing
            )
            z_slice = int(round(voxel[2]))

            if z_slice < 0 or z_slice >= depth:
                continue 

            radius = (d_mm / spacing[0]) / 2.0

            x_center = voxel[0]
            y_center = voxel[1]

            xc = x_center / width
            yc = y_center / height
            w = (radius * 2) / width
            h = (radius * 2) / height

            label_line = f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"

            slice_img_name = f"{scan_id}_{z_slice:04d}.png"
            slice_label_name = f"{scan_id}_{z_slice:04d}.txt"
            slice_label_path = os.path.join(label_out_dir, slice_label_name)

            with open(slice_label_path, "a") as f:
                f.write(label_line)

    print("\n DONE! YOLO labels created.")
    print(f"Saved under: {OUT_LABEL_DIR}")

if __name__ == "__main__":
    create_all_labels()