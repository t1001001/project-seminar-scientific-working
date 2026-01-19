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

def test_cyclegan():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating synthetic CT images using CycleGAN...")
    results_root = os.path.join(conf.ROOT, "results")
    cmd = [
        sys.executable, CYCLEGAN_REPO,
        "--dataroot", DATASET,
        "--name", NAME,
        "--model", "cycle_gan",
        "--phase", "test",
        "--no_dropout",
        "--epoch", str(EPOCHS),
        "--checkpoints_dir", os.path.join(conf.ROOT.parent, "checkpoints"),
        "--results_dir", results_root,
    ]
    subprocess.run(cmd, check=True)
    results_dir = Path(results_root) / NAME / f"test_{EPOCHS}" / "images"
    print(f"Looking for outputs in: {results_dir.resolve()}")
    if not results_dir.exists():
        raise RuntimeError(f"Results directory not found: {results_dir.resolve()}")
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
