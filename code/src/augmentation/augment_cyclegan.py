import os
import subprocess
from pathlib import Path
import utils.config as conf

DATASET = f"{conf.ROOT}/data/cyclegan"
NAME = "luna_cyclegan"
OUT_DIR = Path(f"{conf.ROOT}/data/cyclegan/generated")
CYCLEGAN_REPO = os.path.join(conf.ROOT, "pytorch-CycleGAN-and-pix2pix", "test.py")

def augment_cyclegan():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic CT images using CycleGAN...")
    cmd = [
        "python", CYCLEGAN_REPO,
        "--dataroot", DATASET,
        "--name", NAME,
        "--model", "cycle_gan",
        "--phase", "test",
        "--no_dropout"
    ]
    subprocess.run(cmd, check=True)
    results_dir = Path(f"pytorch-CycleGAN-and-pix2pix/results/{NAME}/test_latest/images")
    print("Copying generated images...")
    for img in results_dir.glob("*.png"):
        target = OUT_DIR / img.name
        img.replace(target)
    print("Synthetic images saved to data/cyclegan/generated!")

if __name__ == "__main__":
    augment_cyclegan()
