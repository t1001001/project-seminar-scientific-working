import os
from pathlib import Path
import utils.config as conf

LABEL_ROOT = Path(f"{conf.ROOT}/data/preprocessed/labels")

n_pos, n_neg, n_boxes = 0, 0, 0
for scan_dir in LABEL_ROOT.iterdir():
    if not scan_dir.is_dir(): continue
    for lbl in scan_dir.glob("*.txt"):
        if os.path.getsize(lbl) == 0:
            n_neg += 1
        else:
            n_pos += 1
            with open(lbl) as f:
                n_boxes += sum(1 for _ in f)

print(n_pos)
print(n_neg)
print(n_boxes)