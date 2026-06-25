import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import (
    DATA_DOCUMENT_MODE,
    DATA_PATH,
    TRAIN_MASK_PATH,
    TRAIN_TOKENS_PATH,
    USE_LOSS_MASK,
    VAL_MASK_PATH,
    VAL_TOKENS_PATH,
    VOCAB_SIZE,
)
from utils import tokenizer

VAL_RATIO = 0.05
COPY_CHUNK_SIZE = 1_000_000
PROGRESS_LINES = 10_000


def token_dtype() -> np.dtype:
    return np.uint16 if VOCAB_SIZE <= np.iinfo(np.uint16).max else np.uint32


def iter_documents():
    with open(DATA_PATH, encoding="utf-8") as src:
        if DATA_DOCUMENT_MODE == "line":
            for line in src:
                text = line.strip()
                if text:
                    yield text
            return

        if DATA_DOCUMENT_MODE != "blankline":
            raise ValueError(f"unknown DATA_DOCUMENT_MODE: {DATA_DOCUMENT_MODE}")

        lines = []
        for line in src:
            text = line.rstrip("\n")
            if text.strip():
                lines.append(text)
            elif lines:
                yield "\n".join(lines).strip()
                lines = []

        if lines:
            yield "\n".join(lines).strip()


def assistant_loss_mask(ids: list[int]) -> list[int]:
    sp = tokenizer.load()
    assistant_id = sp.piece_to_id("<|assistant|>")
    user_id = sp.piece_to_id("<|user|>")
    end_id = sp.piece_to_id("<|end|>")
    eos = tokenizer.eos_id()

    for piece, piece_id in (
        ("<|assistant|>", assistant_id),
        ("<|user|>", user_id),
        ("<|end|>", end_id),
    ):
        if piece_id == tokenizer.unk_id():
            raise ValueError(f"tokenizer is missing special token {piece!r}; rerun training/train_bpe.py")

    learn = False
    mask = []
    for token_id in ids:
        if token_id == user_id:
            learn = False
            mask.append(0)
        elif token_id == assistant_id:
            learn = True
            mask.append(1)
        elif token_id == end_id:
            mask.append(1 if learn else 0)
            learn = False
        elif token_id == eos:
            mask.append(0)
        else:
            mask.append(1 if learn else 0)

    return mask


def write_streamed_tokens(temp_path: Path, mask_temp_path: Path | None, dtype: np.dtype) -> int:
    token_count = 0
    eos = tokenizer.eos_id()

    with open(temp_path, "wb") as dst:
        mask_dst = open(mask_temp_path, "wb") if mask_temp_path is not None else None
        try:
            for doc_idx, text in enumerate(iter_documents(), start=1):
                ids = tokenizer.encode(text)
                if ids:
                    ids = ids + [eos]
                    arr = np.asarray(ids, dtype=dtype)
                    arr.tofile(dst)
                    token_count += len(arr)

                    if mask_dst is not None:
                        mask = np.asarray(assistant_loss_mask(ids), dtype=np.uint8)
                        mask.tofile(mask_dst)

                if doc_idx % PROGRESS_LINES == 0:
                    print(f"encoded {doc_idx} documents | {token_count} tokens")
        finally:
            if mask_dst is not None:
                mask_dst.close()

    return token_count


def copy_memmap(src: np.memmap, dst_path: str, start: int, end: int, dtype: np.dtype) -> None:
    out = np.lib.format.open_memmap(dst_path, mode="w+", dtype=dtype, shape=(end - start,))
    for offset in range(start, end, COPY_CHUNK_SIZE):
        chunk_end = min(offset + COPY_CHUNK_SIZE, end)
        out[offset - start : chunk_end - start] = src[offset:chunk_end]
    out.flush()


def copy_mask_files(mask_temp_path: Path, token_count: int, split_idx: int) -> None:
    if TRAIN_MASK_PATH is None or VAL_MASK_PATH is None:
        raise ValueError("TRAIN_MASK_PATH and VAL_MASK_PATH must be set when USE_LOSS_MASK=True")

    masks = np.memmap(mask_temp_path, mode="r", dtype=np.uint8, shape=(token_count,))
    copy_memmap(masks, TRAIN_MASK_PATH, 0, split_idx, np.uint8)
    copy_memmap(masks, VAL_MASK_PATH, split_idx, token_count, np.uint8)
    del masks


def encode_data() -> None:
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size()} does not match config VOCAB_SIZE {VOCAB_SIZE}; "
            "rerun training/train_bpe.py"
        )

    dtype = token_dtype()
    temp_path = Path(TRAIN_TOKENS_PATH).with_suffix(".tokens.tmp")
    mask_temp_path = Path(TRAIN_TOKENS_PATH).with_suffix(".mask.tmp") if USE_LOSS_MASK else None

    print(f"encoding documents from {DATA_PATH}")
    print(f"writing train tokens to {TRAIN_TOKENS_PATH}")
    print(f"writing val tokens to {VAL_TOKENS_PATH}")
    token_count = write_streamed_tokens(temp_path, mask_temp_path, dtype)
    split_idx = int(token_count * (1 - VAL_RATIO))

    tokens = np.memmap(temp_path, mode="r", dtype=dtype, shape=(token_count,))
    copy_memmap(tokens, TRAIN_TOKENS_PATH, 0, split_idx, dtype)
    copy_memmap(tokens, VAL_TOKENS_PATH, split_idx, token_count, dtype)
    del tokens
    temp_path.unlink(missing_ok=True)

    if mask_temp_path is not None:
        copy_mask_files(mask_temp_path, token_count, split_idx)
        mask_temp_path.unlink(missing_ok=True)
        print(f"train_mask: ({split_idx},)")
        print(f"val_mask: ({token_count - split_idx},)")

    print(f"train: ({split_idx},)")
    print(f"val: ({token_count - split_idx},)")


if __name__ == "__main__":
    encode_data()
