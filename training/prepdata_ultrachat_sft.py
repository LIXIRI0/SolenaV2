import os
import re
import sys
import dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
dotenv.load_dotenv(ROOT_DIR / ".env")

from config import PRETRAIN_HF_TIMEOUT, SFT_DATA_PATH

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(PRETRAIN_HF_TIMEOUT)
os.environ["HF_HUB_ETAG_TIMEOUT"] = str(PRETRAIN_HF_TIMEOUT)

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SPLIT = "train_sft"
DEFAULT_LIMIT = None
FACE_TOKEN = os.getenv("FACE_TOKEN")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_message(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def format_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        content = normalize_message(message.get("content", ""))
        if not content:
            continue

        if role == "user":
            lines.extend(["<|user|>", content])
        elif role == "assistant":
            lines.extend(["<|assistant|>", content])
        else:
            lines.extend([f"<|{role}|>", content])

    lines.append("<|end|>")
    return "\n".join(lines)


def prepare_data(
    output_path: str = SFT_DATA_PATH,
    split: str = DEFAULT_SPLIT,
    limit: int | None = DEFAULT_LIMIT,
    seed: int = 42,
    shuffle: bool = True,
) -> None:
    repo_files = list_repo_files(DATASET_NAME, repo_type="dataset", token=FACE_TOKEN)
    shard_files = sorted(
        file for file in repo_files if file.startswith(f"data/{split}-") and file.endswith(".parquet")
    )
    if not shard_files:
        raise ValueError(f"No parquet shards found for split {split!r}")

    shard_paths = [
        hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=shard_file,
            token=FACE_TOKEN,
        )
        for shard_file in shard_files
    ]

    dataset = load_dataset("parquet", data_files=shard_paths, split="train")

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            f.write(format_messages(example["messages"]))
            f.write("\n\n")
            written += 1

    print(f"Wrote {written} conversations to {output_path}")


if __name__ == "__main__":
    prepare_data()
