import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import DATA_PATH
from utils import tokenizer

VAL_RATIO = 0.05


with open(DATA_PATH, encoding="utf-8") as f:
    ids = tokenizer.encode(f.read(), eos=True)

arr = np.array(ids, dtype=np.uint16)
split_idx = int(len(arr) * (1 - VAL_RATIO))

train_arr = arr[:split_idx]
val_arr = arr[split_idx:]

np.save(ROOT_DIR / "data" / "train.npy", train_arr)
np.save(ROOT_DIR / "data" / "val.npy", val_arr)

print(f"train: {train_arr.shape}")
print(f"val: {val_arr.shape}")
