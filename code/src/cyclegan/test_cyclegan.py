import os
import sys
import subprocess
from pathlib import Path
import utils.config as conf

DATASET = f"{conf.ROOT}/data/cyclegan"
NAME = "luna_cyclegan"
EPOCHS = 50
OUT_DIR = Path(f"{conf.ROOT}/data/cyclegan/generated")
CYCLEGAN_REPO = os.path.join(conf.ROOT, "pytorch-CycleGAN-and-pix2pix", "test.py")

def test_cyclegan():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic CT images using CycleGAN...")
    cmd = [
        sys.executable, CYCLEGAN_REPO,
        "--dataroot", DATASET,
        "--name", NAME,
        "--model", "cycle_gan",
        "--phase", "test",
        "--no_dropout",
        "--epoch", EPOCHS,
        "--checkpoints_dir", os.path.join(conf.ROOT.parent, "checkpoints"),
    ]
    subprocess.run(cmd, check=True)

    results_dir = Path(f"pytorch-CycleGAN-and-pix2pix/results/{NAME}/test_latest/images")
    print("Copying generated images...")
    count = 0
    for img in results_dir.glob("*.png"):
        target = OUT_DIR / img.name
        img.replace(target)
        count += 1

    if count == 0:
        raise RuntimeError("No CycleGAN outputs found in results directory.")

    print(f"Synthetic images saved to data/cyclegan/generated ({count} files).")

if __name__ == "__main__":
    test_cyclegan()
