import os
import sys
import subprocess
import utils.config as conf

DATASET = os.path.join(conf.ROOT, "data", "cyclegan")
NAME = "luna_cyclegan"
EPOCHS = 50
CYCLEGAN_REPO = os.path.join(conf.ROOT, "pytorch-CycleGAN-and-pix2pix", "train.py")

def train_cyclegan():
    print("Starting CycleGAN training...")
    cmd = [
        sys.executable,            # nutzt den aktiven venv-Interpreter
        CYCLEGAN_REPO,             # direkt das Skript, kein zusätzliches 'python'
        "--dataroot", DATASET,
        "--name", NAME,
        "--model", "cycle_gan",
        "--n_epochs", str(EPOCHS),
        "--n_epochs_decay", str(EPOCHS),
        "--max_dataset_size", "2000",
        "--num_threads", "0",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("CycleGAN training complete!")

if __name__ == "__main__":
    train_cyclegan()
