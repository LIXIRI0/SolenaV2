import hashlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

import dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
dotenv.load_dotenv(ROOT_DIR / ".env")

from config import (
    PRETRAIN_DATA_PATH,
    PRETRAIN_CHARS_PER_TOKEN,
    PRETRAIN_HF_TIMEOUT,
    PRETRAIN_MIX,
    PRETRAIN_SEED,
    PRETRAIN_SHUFFLE_BUFFER_SIZE,
    PRETRAIN_TARGET_TOKENS,
)

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(PRETRAIN_HF_TIMEOUT)
os.environ["HF_HUB_ETAG_TIMEOUT"] = str(PRETRAIN_HF_TIMEOUT)
os.environ["HF_DATASETS_DOWNLOAD_TIMEOUT"] = str(PRETRAIN_HF_TIMEOUT)
os.environ["HF_DATASETS_ETAG_TIMEOUT"] = str(PRETRAIN_HF_TIMEOUT)

from datasets import load_dataset
from utils.gcs_cache import sync_raw_data_to_gcs

FACE_TOKEN = os.getenv("FACE_TOKEN")
MIN_DOC_CHARS = 200
DEFAULT_MAX_CHARS = 16_000
CS_MAX_CHARS = 12_000
PROGRESS_DOCS = 10_000
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class DatasetOption:
    dataset: str
    config: str | None
    split: str
    field_candidates: tuple[str, ...]
    data_dir: str | None = None
    streaming: bool = True


@dataclass(frozen=True)
class SourceSpec:
    name: str
    weight: float
    options: tuple[DatasetOption, ...]
    max_chars: int = DEFAULT_MAX_CHARS


SOURCES = (
    SourceSpec(
        name="clean_web",
        weight=PRETRAIN_MIX["clean_web"],
        options=(
            DatasetOption(
                dataset="HuggingFaceTB/smollm-corpus",
                config="fineweb-edu-dedup",
                split="train",
                field_candidates=("text",),
            ),
            DatasetOption(
                dataset="HuggingFaceFW/fineweb-edu",
                config="sample-10BT",
                split="train",
                field_candidates=("text",),
            ),
        ),
    ),
    SourceSpec(
        name="wiki",
        weight=PRETRAIN_MIX["wiki"],
        options=(
            DatasetOption(
                dataset="Salesforce/wikitext",
                config="wikitext-103-raw-v1",
                split="train",
                field_candidates=("text",),
            ),
            DatasetOption(
                dataset="wikimedia/wikipedia",
                config="20231101.en",
                split="train",
                field_candidates=("text",),
            ),
        ),
    ),
    SourceSpec(
        name="textbooks",
        weight=PRETRAIN_MIX["textbooks"],
        options=(
            DatasetOption(
                dataset="HuggingFaceTB/smollm-corpus",
                config="cosmopedia-v2",
                split="train",
                field_candidates=("text", "prompt"),
            ),
        ),
    ),
    SourceSpec(
        name="stories",
        weight=PRETRAIN_MIX["stories"],
        options=(
            DatasetOption(
                dataset="roneneldan/TinyStories",
                config=None,
                split="train",
                field_candidates=("text",),
            ),
            DatasetOption(
                dataset="emozilla/pg19",
                config=None,
                split="train",
                field_candidates=("text", "content"),
            ),
        ),
    ),
    SourceSpec(
        name="cs",
        weight=PRETRAIN_MIX["cs"],
        max_chars=CS_MAX_CHARS,
        options=(
            DatasetOption(
                dataset="bigcode/the-stack-smol-xl",
                config=None,
                split="train",
                field_candidates=("content",),
            ),
            DatasetOption(
                dataset="ml6team/the-stack-smol-python",
                config=None,
                split="train",
                field_candidates=("content",),
            ),
        ),
    ),
)


def stringify_value(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = [stringify_value(item) for item in value.values()]
        return " ".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        parts = [stringify_value(item) for item in value]
        return " ".join(part for part in parts if part)
    return None


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_text(example: dict, field_candidates: tuple[str, ...]) -> str | None:
    for field in field_candidates:
        value = stringify_value(example.get(field))
        if value and value.strip():
            return value
    return None


def chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    for start in range(0, len(text), max_chars):
        chunk = text[start : start + max_chars].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_stream(option: DatasetOption):
    kwargs = {
        "split": option.split,
        "streaming": option.streaming,
        "token": FACE_TOKEN,
    }
    if option.data_dir is not None:
        kwargs["data_dir"] = option.data_dir
    if option.config is None:
        return load_dataset(option.dataset, **kwargs)
    return load_dataset(option.dataset, option.config, **kwargs)


def iter_source_examples(source: SourceSpec):
    last_error: Exception | None = None
    for option in source.options:
        try:
            option_name = option.dataset
            if option.config is not None:
                option_name = f"{option_name}/{option.config}"
            if option.data_dir is not None:
                option_name = f"{option_name}/{option.data_dir}"
            print(f"{source.name}: opening {option_name}", flush=True)
            stream = load_stream(option)
            if PRETRAIN_SHUFFLE_BUFFER_SIZE > 0:
                print(
                    f"{source.name}: shuffling stream with buffer {PRETRAIN_SHUFFLE_BUFFER_SIZE}",
                    flush=True,
                )
                stream = stream.shuffle(seed=PRETRAIN_SEED, buffer_size=PRETRAIN_SHUFFLE_BUFFER_SIZE)
            for example in stream:
                yield option, example
            print(f"{source.name}: exhausted {option.dataset}; trying fallback if available", flush=True)
        except Exception as exc:
            last_error = exc
            print(f"{source.name}: failed {option.dataset}; trying fallback if available ({exc})", flush=True)

    if last_error is not None:
        raise RuntimeError(f"all dataset options failed for source {source.name}") from last_error


def iter_source_documents(
    source: SourceSpec,
    seen_hashes: set[str],
    stats: dict[str, int],
) -> Iterator[str]:
    for option, example in iter_source_examples(source):
        raw_text = extract_text(example, option.field_candidates)
        if raw_text is None:
            stats["skipped"] += 1
            continue

        text = normalize_text(raw_text)
        if len(text) < MIN_DOC_CHARS:
            stats["skipped"] += 1
            continue

        for chunk in chunk_text(text, source.max_chars):
            if len(chunk) < MIN_DOC_CHARS:
                stats["skipped"] += 1
                continue

            text_hash = stable_hash(chunk)
            if text_hash in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(text_hash)

            yield chunk


def empty_stats() -> dict[str, int]:
    return {
        "docs": 0,
        "chars": 0,
        "skipped": 0,
        "duplicates": 0,
    }


@dataclass
class SourceState:
    source: SourceSpec
    char_budget: int
    iterator: Iterator[str]
    stats: dict[str, int]
    exhausted: bool = False

    @property
    def remaining_chars(self) -> int:
        return max(self.char_budget - self.stats["chars"], 0)

    @property
    def active(self) -> bool:
        return not self.exhausted and self.remaining_chars > 0


def write_sources(
    output,
    states: list[SourceState],
) -> None:
    for state in states:
        while state.active:
            try:
                chunk = next(state.iterator)
            except StopIteration:
                state.exhausted = True
                print(f"{state.source.name}: stream exhausted before target budget")
                break

            output.write(chunk.replace("\n", " ") + "\n")
            state.stats["docs"] += 1
            state.stats["chars"] += len(chunk)

            if state.stats["docs"] % PROGRESS_DOCS == 0:
                approx_tokens = state.stats["chars"] // PRETRAIN_CHARS_PER_TOKEN
                print(
                    f"{state.source.name}: {state.stats['docs']} docs | "
                    f"{state.stats['chars']} chars | ~{approx_tokens} tokens",
                    flush=True,
                )


def validate_mix() -> None:
    total = sum(PRETRAIN_MIX.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"PRETRAIN_MIX weights must sum to 1.0, got {total}")

    source_names = {source.name for source in SOURCES}
    mix_names = set(PRETRAIN_MIX)
    if source_names != mix_names:
        raise ValueError(f"PRETRAIN_MIX keys {mix_names} do not match source names {source_names}")


def prepare_data(output_path: str = PRETRAIN_DATA_PATH) -> None:
    validate_mix()
    target_chars = PRETRAIN_TARGET_TOKENS * PRETRAIN_CHARS_PER_TOKEN
    seen_hashes: set[str] = set()
    states = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output:
        for source in SOURCES:
            char_budget = int(target_chars * source.weight)
            print(
                f"{source.name}: target {char_budget} chars "
                f"(~{char_budget // PRETRAIN_CHARS_PER_TOKEN} tokens)",
                flush=True,
            )
            stats = empty_stats()
            states.append(
                SourceState(
                    source=source,
                    char_budget=char_budget,
                    iterator=iter_source_documents(source, seen_hashes, stats),
                    stats=stats,
                )
            )

        write_sources(output, states)

    all_stats = {state.source.name: state.stats for state in states}
    total_chars = sum(stats["chars"] for stats in all_stats.values())
    print(f"Wrote mixed pretraining corpus to {output_path}")
    for name, stats in all_stats.items():
        share = (stats["chars"] / total_chars) if total_chars else 0.0
        approx_tokens = stats["chars"] // PRETRAIN_CHARS_PER_TOKEN
        print(
            f"{name}: docs={stats['docs']} chars={stats['chars']} "
            f"~tokens={approx_tokens} share={share:.2%} "
            f"skipped={stats['skipped']} duplicates={stats['duplicates']}"
        )
    sync_raw_data_to_gcs()


if __name__ == "__main__":
    prepare_data()
