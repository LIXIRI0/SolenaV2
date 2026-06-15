import os
import sys
import dotenv
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
dotenv.load_dotenv(ROOT_DIR / ".env")

from config import DATA_PATH

DATASET_NAME = "HuggingFaceH4/ultrachat_200k"
DEFAULT_SPLIT = "train_sft"
DEFAULT_LIMIT = 5000
FACE_TOKEN = os.getenv("FACE_TOKEN")


def format_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "").strip()
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
    output_path: str = DATA_PATH,
    split: str = DEFAULT_SPLIT,
    limit: int = DEFAULT_LIMIT,
    seed: int = 42,
    shuffle: bool = True,
) -> None:
    repo_files = list_repo_files(DATASET_NAME, repo_type="dataset", token=FACE_TOKEN)
    shard_files = sorted(
        file for file in repo_files if file.startswith(f"data/{split}-") and file.endswith(".parquet")
    )
    if not shard_files:
        raise ValueError(f"No parquet shards found for split {split!r}")

    shard_path = hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=shard_files[0],
        token=FACE_TOKEN,
    )

    dataset = load_dataset("parquet", data_files=shard_path, split="train")

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    dataset = dataset.select(range(min(limit, len(dataset))))
    texts = [format_messages(example["messages"]) for example in dataset]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts))
        f.write("\n")

    print(f"Wrote {len(texts)} conversations to {output_path}")



if __name__ == "__main__":
    prepare_data()
