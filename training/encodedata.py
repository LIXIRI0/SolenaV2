import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import DATA_PATH, TRAIN_TOKENS_PATH, VAL_TOKENS_PATH, VOCAB_SIZE
from utils import tokenizer

VAL_RATIO = 0.05
COPY_CHUNK_SIZE = 1_000_000
PROGRESS_LINES = 10_000


def token_dtype() -> np.dtype:
    return np.uint16 if VOCAB_SIZE <= np.iinfo(np.uint16).max else np.uint32


def write_streamed_tokens(temp_path: Path, dtype: np.dtype) -> int:
    token_count = 0
    eos = tokenizer.eos_id()

    with open(DATA_PATH, encoding="utf-8") as src, open(temp_path, "wb") as dst:
        for line_idx, line in enumerate(src, start=1):
            ids = tokenizer.encode(line)
            if ids:
                arr = np.asarray(ids, dtype=dtype)
                arr.tofile(dst)
                token_count += len(arr)

            if line_idx % PROGRESS_LINES == 0:
                print(f"encoded {line_idx} lines | {token_count} tokens")

        np.asarray([eos], dtype=dtype).tofile(dst)
        token_count += 1

    return token_count


def copy_memmap(src: np.memmap, dst_path: str, start: int, end: int, dtype: np.dtype) -> None:
    out = np.lib.format.open_memmap(dst_path, mode="w+", dtype=dtype, shape=(end - start,))
    for offset in range(start, end, COPY_CHUNK_SIZE):
        chunk_end = min(offset + COPY_CHUNK_SIZE, end)
        out[offset - start : chunk_end - start] = src[offset:chunk_end]
    out.flush()


def encode_data() -> None:
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size()} does not match config VOCAB_SIZE {VOCAB_SIZE}; "
            "rerun training/train_bpe.py"
        )

    dtype = token_dtype()
    temp_path = Path(TRAIN_TOKENS_PATH).with_suffix(".tokens.tmp")

    token_count = write_streamed_tokens(temp_path, dtype)
    split_idx = int(token_count * (1 - VAL_RATIO))

    tokens = np.memmap(temp_path, mode="r", dtype=dtype, shape=(token_count,))
    copy_memmap(tokens, TRAIN_TOKENS_PATH, 0, split_idx, dtype)
    copy_memmap(tokens, VAL_TOKENS_PATH, split_idx, token_count, dtype)
    del tokens
    temp_path.unlink(missing_ok=True)

    print(f"train: ({split_idx},)")
    print(f"val: ({token_count - split_idx},)")


if __name__ == "__main__":
    encode_data()
