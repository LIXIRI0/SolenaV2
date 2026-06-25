import hashlib
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
    VAL_RATIO,
    VAL_MASK_PATH,
    VAL_TOKENS_PATH,
    VOCAB_SIZE,
)
from utils import tokenizer

COPY_CHUNK_SIZE = 1_000_000
PROGRESS_LINES = 10_000
SPLIT_HASH_SEED = "solena-v2-split-v1"


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


def use_val_split(text: str) -> bool:
    digest = hashlib.blake2b(
        f"{SPLIT_HASH_SEED}\0{text}".encode("utf-8"),
        digest_size=8,
    ).digest()
    bucket = int.from_bytes(digest, "big") / 2**64
    return bucket < VAL_RATIO


def write_ids(dst, mask_dst, ids: list[int], dtype: np.dtype) -> int:
    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(dst)

    if mask_dst is not None:
        mask = np.asarray(assistant_loss_mask(ids), dtype=np.uint8)
        mask.tofile(mask_dst)

    return len(arr)


def write_streamed_split_tokens(
    train_temp_path: Path,
    val_temp_path: Path,
    train_mask_temp_path: Path | None,
    val_mask_temp_path: Path | None,
    dtype: np.dtype,
) -> dict[str, int]:
    stats = {
        "train_docs": 0,
        "val_docs": 0,
        "train_tokens": 0,
        "val_tokens": 0,
    }
    eos = tokenizer.eos_id()

    with open(train_temp_path, "wb") as train_dst, open(val_temp_path, "wb") as val_dst:
        train_mask_dst = open(train_mask_temp_path, "wb") if train_mask_temp_path is not None else None
        val_mask_dst = open(val_mask_temp_path, "wb") if val_mask_temp_path is not None else None
        try:
            for doc_idx, text in enumerate(iter_documents(), start=1):
                ids = tokenizer.encode(text)
                if ids:
                    ids = ids + [eos]
                    if use_val_split(text):
                        stats["val_tokens"] += write_ids(val_dst, val_mask_dst, ids, dtype)
                        stats["val_docs"] += 1
                    else:
                        stats["train_tokens"] += write_ids(train_dst, train_mask_dst, ids, dtype)
                        stats["train_docs"] += 1

                if doc_idx % PROGRESS_LINES == 0:
                    print(
                        f"encoded {doc_idx} documents | "
                        f"train={stats['train_tokens']} tokens | val={stats['val_tokens']} tokens"
                    )
        finally:
            if train_mask_dst is not None:
                train_mask_dst.close()
            if val_mask_dst is not None:
                val_mask_dst.close()

    return stats


def copy_memmap(src: np.memmap, dst_path: str, start: int, end: int, dtype: np.dtype) -> None:
    out = np.lib.format.open_memmap(dst_path, mode="w+", dtype=dtype, shape=(end - start,))
    for offset in range(start, end, COPY_CHUNK_SIZE):
        chunk_end = min(offset + COPY_CHUNK_SIZE, end)
        out[offset - start : chunk_end - start] = src[offset:chunk_end]
    out.flush()
    del out


def copy_temp_to_staged_npy(temp_path: Path, output_path: str, token_count: int, dtype: np.dtype) -> Path:
    staged_path = Path(f"{output_path}.next.npy")
    tokens = np.memmap(temp_path, mode="r", dtype=dtype, shape=(token_count,))
    copy_memmap(tokens, str(staged_path), 0, token_count, dtype)
    del tokens
    return staged_path


def replace_staged_outputs(staged_outputs: list[tuple[Path, str]]) -> None:
    for staged_path, output_path in staged_outputs:
        staged_path.replace(output_path)


def encode_data() -> None:
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size()} does not match config VOCAB_SIZE {VOCAB_SIZE}; "
            "rerun training/train_bpe.py"
        )

    dtype = token_dtype()
    train_temp_path = Path(TRAIN_TOKENS_PATH).with_suffix(".tokens.tmp")
    val_temp_path = Path(VAL_TOKENS_PATH).with_suffix(".tokens.tmp")
    train_mask_temp_path = Path(TRAIN_TOKENS_PATH).with_suffix(".mask.tmp") if USE_LOSS_MASK else None
    val_mask_temp_path = Path(VAL_TOKENS_PATH).with_suffix(".mask.tmp") if USE_LOSS_MASK else None

    print(f"encoding documents from {DATA_PATH}")
    print(f"writing train tokens to {TRAIN_TOKENS_PATH}")
    print(f"writing val tokens to {VAL_TOKENS_PATH}")
    stats = write_streamed_split_tokens(
        train_temp_path,
        val_temp_path,
        train_mask_temp_path,
        val_mask_temp_path,
        dtype,
    )

    train_tokens = stats["train_tokens"]
    val_tokens = stats["val_tokens"]
    if train_tokens == 0 or val_tokens == 0:
        raise ValueError(
            f"empty split after document hash split: train={train_tokens}, val={val_tokens}; "
            "use more source documents or adjust VAL_RATIO"
        )

    staged_outputs = [
        (copy_temp_to_staged_npy(train_temp_path, TRAIN_TOKENS_PATH, train_tokens, dtype), TRAIN_TOKENS_PATH),
        (copy_temp_to_staged_npy(val_temp_path, VAL_TOKENS_PATH, val_tokens, dtype), VAL_TOKENS_PATH),
    ]

    if train_mask_temp_path is not None and val_mask_temp_path is not None:
        if TRAIN_MASK_PATH is None or VAL_MASK_PATH is None:
            raise ValueError("TRAIN_MASK_PATH and VAL_MASK_PATH must be set when USE_LOSS_MASK=True")

        staged_outputs.extend(
            [
                (
                    copy_temp_to_staged_npy(train_mask_temp_path, TRAIN_MASK_PATH, train_tokens, np.uint8),
                    TRAIN_MASK_PATH,
                ),
                (
                    copy_temp_to_staged_npy(val_mask_temp_path, VAL_MASK_PATH, val_tokens, np.uint8),
                    VAL_MASK_PATH,
                ),
            ]
        )
        print(f"train_mask: ({train_tokens},)")
        print(f"val_mask: ({val_tokens},)")

    replace_staged_outputs(staged_outputs)
    train_temp_path.unlink(missing_ok=True)
    val_temp_path.unlink(missing_ok=True)
    if train_mask_temp_path is not None:
        train_mask_temp_path.unlink(missing_ok=True)
    if val_mask_temp_path is not None:
        val_mask_temp_path.unlink(missing_ok=True)

    total_tokens = train_tokens + val_tokens
    val_share = val_tokens / total_tokens
    print(f"train: ({train_tokens},) docs={stats['train_docs']}")
    print(f"val: ({val_tokens},) docs={stats['val_docs']} share={val_share:.2%}")


if __name__ == "__main__":
    encode_data()
