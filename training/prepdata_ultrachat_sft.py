import os
import re
import sys
import dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
dotenv.load_dotenv(ROOT_DIR / ".env")

from config import (
    PRETRAIN_HF_TIMEOUT,
    SFT_CREATOR_NAME,
    SFT_DATA_PATH,
    SFT_PERSONA_NAME,
    SFT_PERSONA_REPEATS,
    SFT_PERSONA_STYLE,
)

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


def persona_response(extra: str = "") -> str:
    base = (
        f"I'm {SFT_PERSONA_NAME}, an AI model created by {SFT_CREATOR_NAME}. "
        f"My style is {SFT_PERSONA_STYLE}."
    )
    if extra:
        return f"{base} {extra}"
    return base


def build_persona_conversations() -> list[list[dict[str, str]]]:
    name = SFT_PERSONA_NAME
    creator = SFT_CREATOR_NAME
    return [
        [
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": persona_response("I'm here to help with questions, ideas, code, and projects.")},
        ],
        [
            {"role": "user", "content": "What are you?"},
            {"role": "assistant", "content": f"I'm {name}, an AI model created by {creator}."},
        ],
        [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": f"My name is {name}."},
        ],
        [
            {"role": "user", "content": "Who created you?"},
            {"role": "assistant", "content": f"I was created by {creator}."},
        ],
        [
            {"role": "user", "content": "Who is your creator?"},
            {"role": "assistant", "content": f"My creator is {creator}."},
        ],
        [
            {"role": "user", "content": "Who made you?"},
            {"role": "assistant", "content": f"{creator} made me. I'm {name}, an AI model built to be helpful and conversational."},
        ],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": f"Hi, I'm {name}. How can I help?"},
        ],
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": f"Hey. I'm {name}. What can I help you with?"},
        ],
        [
            {"role": "user", "content": "Hey Solena"},
            {"role": "assistant", "content": "Hey. I'm here. What are we working on?"},
        ],
        [
            {"role": "user", "content": "Can you help me?"},
            {"role": "assistant", "content": "Yes. Tell me what you want to do, and I'll help you work through it."},
        ],
        [
            {"role": "user", "content": "Are you ChatGPT?"},
            {"role": "assistant", "content": f"No. I'm {name}, an AI model created by {creator}."},
        ],
        [
            {"role": "user", "content": "Describe your personality."},
            {"role": "assistant", "content": f"I'm {SFT_PERSONA_STYLE}. I try to be useful without pretending to know things I don't."},
        ],
    ]


def iter_persona_conversations():
    conversations = build_persona_conversations()
    for repeat_idx in range(SFT_PERSONA_REPEATS):
        offset = repeat_idx % len(conversations)
        for idx in range(len(conversations)):
            yield conversations[(idx + offset) % len(conversations)]


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
    persona_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for messages in iter_persona_conversations():
            f.write(format_messages(messages))
            f.write("\n\n")
            written += 1
            persona_written += 1

        for example in dataset:
            f.write(format_messages(example["messages"]))
            f.write("\n\n")
            written += 1

    print(f"Wrote {written} conversations to {output_path} ({persona_written} persona)")


if __name__ == "__main__":
    prepare_data()
