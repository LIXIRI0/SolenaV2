from dataclasses import dataclass

import numpy as np

from config import BATCH_SIZE, NUM_DEVICES, PER_DEVICE_BATCH_SIZE, SEQ_LEN, TRAIN_TOKENS_PATH, VAL_TOKENS_PATH


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

    def split_data(self, split: str) -> np.ndarray:
        if split == "train":
            return self.train
        if split == "val":
            return self.val
        raise ValueError(f"unknown split: {split}")

    def batch_from_starts(self, data: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        starts = np.asarray(starts, dtype=np.int64)

        x = np.stack([data[i : i + self.seq_len] for i in starts])
        y = np.stack([data[i + 1 : i + self.seq_len + 1] for i in starts])

        return x.astype(np.int32), y.astype(np.int32)

    def get_batch(self, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
        data = self.split_data(split)
        starts = np.random.randint(0, len(data) - self.seq_len - 1, size=self.batch_size)
        return self.batch_from_starts(data, starts)

    def get_train_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_batch("train")

    def get_val_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_batch("val")

    def get_eval_batch(
        self,
        split: str,
        batch_idx: int,
        num_batches: int,
        batch_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_batches <= 0:
            raise ValueError("num_batches must be > 0")

        data = self.split_data(split)
        batch_size = self.batch_size if batch_size is None else batch_size
        max_start = len(data) - self.seq_len - 1
        total_samples = max(1, num_batches * batch_size)
        stride = max(1, max_start // total_samples)
        sample_ids = batch_idx * batch_size + np.arange(batch_size)
        starts = (sample_ids * stride) % max_start
        return self.batch_from_starts(data, starts)

    def get_val_eval_batch(self, batch_idx: int, num_batches: int) -> tuple[np.ndarray, np.ndarray]:
        return self.get_eval_batch("val", batch_idx, num_batches)

    def get_sharded_batch(
        self,
        split: str = "train",
        num_devices: int = NUM_DEVICES,
        per_device_batch_size: int = PER_DEVICE_BATCH_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        old_batch_size = self.batch_size
        self.batch_size = num_devices * per_device_batch_size
        try:
            x, y = self.get_batch(split)
        finally:
            self.batch_size = old_batch_size

        x = x.reshape(num_devices, per_device_batch_size, self.seq_len)
        y = y.reshape(num_devices, per_device_batch_size, self.seq_len)
        return x, y

    def get_sharded_train_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_sharded_batch("train")

    def get_sharded_val_batch(self) -> tuple[np.ndarray, np.ndarray]:
        return self.get_sharded_batch("val")

    def get_sharded_eval_batch(
        self,
        split: str,
        batch_idx: int,
        num_batches: int,
        num_devices: int = NUM_DEVICES,
        per_device_batch_size: int = PER_DEVICE_BATCH_SIZE,
    ) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.get_eval_batch(
            split,
            batch_idx,
            num_batches,
            batch_size=num_devices * per_device_batch_size,
        )
        x = x.reshape(num_devices, per_device_batch_size, self.seq_len)
        y = y.reshape(num_devices, per_device_batch_size, self.seq_len)
        return x, y

    def get_sharded_val_eval_batch(self, batch_idx: int, num_batches: int) -> tuple[np.ndarray, np.ndarray]:
        return self.get_sharded_eval_batch("val", batch_idx, num_batches)


def load_tokens(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_dataset() -> TokenDataset:
    return TokenDataset(
        train=load_tokens(TRAIN_TOKENS_PATH),
        val=load_tokens(VAL_TOKENS_PATH),
    )
