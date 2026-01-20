import os
import sys
import subprocess
from pathlib import Path
import utils.config as conf

DATASET = os.path.join(conf.ROOT, "data", "cyclegan")
NAME = "luna_cyclegan"
EPOCHS = 25
OUT_DIR = Path(conf.ROOT) / "data" / "cyclegan" / "generated"
CYCLEGAN_REPO = os.path.join(conf.ROOT, "pytorch-CycleGAN-and-pix2pix", "test.py")

def cyclegan_generate():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dataset_dir = Path(conf.ROOT / "data" / "cyclegan" / "temp")
    testA = temp_dataset_dir / "testA"
    testB = temp_dataset_dir / "testB"
    testA.mkdir(parents=True, exist_ok=True)
    testB.mkdir(parents=True, exist_ok=True)
    labeled_images = []
    for img in Path(conf.ROOT / "data" / "preprocessed" / "images").rglob("*.png"):
        uid, idx = img.stem.rsplit("_", 1)
        lbl_file = Path(conf.ROOT / "data" / "preprocessed" / "labels") / uid / f"{uid}_{idx}.txt"
        if lbl_file.exists():
            labeled_images.append(str(img))
            target = testA / img.name
            if not target.exists():
                target.symlink_to(img.resolve())
    if not labeled_images:
        print("No labeled images found for CycleGAN generation")
        return
    if not any(testB.glob("*.png")):
        first_img = next(testA.glob("*.png"))
        dummy_B = testB / first_img.name
        dummy_B.symlink_to(first_img.resolve())
    print(f"Generating synthetic CT images for {len(labeled_images)} labeled images using CycleGAN...")
    results_root = Path(conf.ROOT) / "results"
    cmd = [
        sys.executable, str(Path(conf.ROOT) / "pytorch-CycleGAN-and-pix2pix" / "test.py"),
        "--dataroot", str(temp_dataset_dir),
        "--name", NAME,
        "--model", "cycle_gan",
        "--phase", "test",
        "--no_dropout",
        "--epoch", str(EPOCHS),
        "--checkpoints_dir", str(Path(conf.ROOT).parent / "checkpoints"),
        "--results_dir", str(results_root),
    ]
    subprocess.run(cmd, check=True)
    results_dir = results_root / NAME / f"test_{EPOCHS}" / "images"
    count = 0
    if results_dir.exists():
        for img in results_dir.glob("*.png"):
            target = OUT_DIR / img.name
            img.replace(target)
            count += 1
    print(f"Synthetic images saved to {OUT_DIR} ({count} files).")

if __name__ == "__main__":
    cyclegan_generate()
