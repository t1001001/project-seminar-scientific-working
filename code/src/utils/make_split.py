import json, random
from pathlib import Path
import utils.config as conf

def make_split():
    import json, random
    from pathlib import Path
    import utils.config as conf

    IMG_DIR = Path(f"{conf.ROOT}/data/preprocessed/images")
    TRAIN_RATIO = 0.8
    OUT = Path(f"{conf.ROOT}/data/split.json")

    pairs = []
    for scan in IMG_DIR.iterdir():
        if scan.is_dir():
            for img in scan.glob("*.png"):
                pairs.append(img.name)

    random.seed(42)
    random.shuffle(pairs)

    k = int(len(pairs) * TRAIN_RATIO)
    split = {
        "train": pairs[:k],
        "val": pairs[k:]
    }

    with open(OUT, "w") as f:
        json.dump(split, f, indent=2)

    print("Saved split.json")

if __name__ == "__main__":
    split_path = Path(f"{conf.ROOT}/data/split.json")
    if split_path.exists():
        print("split.json already exists — NOT regenerating.")
    else:
        make_split()
