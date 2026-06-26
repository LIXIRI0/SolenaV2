import os
import sys
from pathlib import Path

import sentencepiece as spm

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import TOKENIZER_PATH, VOCAB_SIZE, PRETRAIN_DATA_PATH
from utils.gcs_cache import sync_pretrain_tokenizer_inputs_from_gcs, sync_tokenizer_to_gcs


def train_tokenizer():
    sync_pretrain_tokenizer_inputs_from_gcs()
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=PRETRAIN_DATA_PATH,
        model_prefix=TOKENIZER_PATH.replace(".model", ""),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        user_defined_symbols=["<|user|>", "<|assistant|>", "<|end|>"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print(f"Tokenizer trained and saved to {TOKENIZER_PATH}")
    sync_tokenizer_to_gcs()

if __name__ == "__main__":
    train_tokenizer()
