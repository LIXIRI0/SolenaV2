from dataclasses import dataclass

import numpy as np

from config import BATCH_SIZE, SEQ_LEN, TRAIN_TOKENS_PATH, VAL_TOKENS_PATH


@dataclass
class TokenDataset:
    train: np.ndarray
    val: np.ndarray
    seq_len: int = SEQ_LEN
    batch_size: int = BATCH_SIZE

    def __post_init__(self) -> None:
        if len(self.train) <= self.seq_len+1:
            raise ValueError("train token array is too short for SEQ_LEN")
        if len(self.val) <= self.seq_len+1:
            raise ValueError("val token array is too short for SEQ_LEN")

    def get_batch(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            data = self.train
        elif split == "val":
            data = self.val
        else:
            raise ValueError(f"unknown split: {split}")

        starts = np.random.randint(0, len(data) - self.seq_len - 1, size=self.batch_size)

        x = np.stack([data[i : i + self.seq_len] for i in starts])
        y = np.stack([data[i + 1 : i + self.seq_len + 1] for i in starts])

        return x.astype(np.int32), y.astype(np.int32)

    def get_train_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_batch("train")

    def get_val_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_batch("val")


def load_tokens(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_dataset() -> TokenDataset:
    return TokenDataset(
        train=load_tokens(TRAIN_TOKENS_PATH),
        val=load_tokens(VAL_TOKENS_PATH),
    )
