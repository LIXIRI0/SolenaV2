import argparse
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

from config import (
    BASE_CHECKPOINT_PATH,
    CHECKPOINT_PATH,
    GEN_EXIT_COMMANDS,
    GEN_MAX_BANNED_TOKENS,
    GEN_MAX_NEW_TOKENS,
    GEN_MIN_NEW_TOKENS,
    GEN_NO_REPEAT_NGRAM_SIZE,
    GEN_PROMPT_MODE,
    GEN_REPETITION_PENALTY,
    GEN_REPETITION_WINDOW,
    GEN_SEED,
    GEN_SHOW_FULL_TEXT,
    GEN_STOP_AFTER_SENTENCE,
    GEN_TEMPERATURE,
    GEN_TOP_K,
    GEN_TOP_P,
    SEQ_LEN,
    VOCAB_SIZE,
)
from models.SolenaV2 import SolenaV2
from utils import tokenizer

CHAT_BOUNDARIES = ("<|user|>", "<|assistant|>", "<|end|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args()


def format_prompt(prompt: str) -> str:
    mode = GEN_PROMPT_MODE.lower()
    if mode == "plain":
        return prompt
    if mode == "chat":
        return f"<|user|>\n{prompt}\n<|assistant|>\n"
    raise ValueError('GEN_PROMPT_MODE must be "plain" or "chat"')


def load_model() -> SolenaV2:
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size()} does not match config VOCAB_SIZE {VOCAB_SIZE}; "
            "rerun training/train_bpe.py and training/encodedata.py"
        )
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"checkpoint not found: {CHECKPOINT_PATH}\n"
            f"active generation mode is {GEN_PROMPT_MODE!r}. "
            f"For SFT generation, copy/download the trained SFT checkpoint there. "
            f"Base checkpoint path is {BASE_CHECKPOINT_PATH}."
        )

    model = SolenaV2(jax.random.PRNGKey(0))
    return eqx.tree_deserialise_leaves(CHECKPOINT_PATH, model)


def apply_top_k(logits: jax.Array, top_k: int | None) -> jax.Array:
    if top_k is None:
        return logits

    top_k = min(top_k, logits.shape[-1])
    threshold = jnp.sort(logits)[-top_k]
    return jnp.where(logits < threshold, -jnp.inf, logits)


def apply_top_p(logits: jax.Array, top_p: float | None) -> jax.Array:
    if top_p is None:
        return logits

    sorted_indices = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    sorted_probs = jax.nn.softmax(sorted_logits)
    cumulative_probs = jnp.cumsum(sorted_probs)

    sorted_remove = cumulative_probs > top_p
    sorted_remove = jnp.concatenate([jnp.array([False]), sorted_remove[:-1]])
    remove = jnp.zeros_like(logits, dtype=bool).at[sorted_indices].set(sorted_remove)
    return jnp.where(remove, -jnp.inf, logits)


def token_mask(logits: jax.Array, ids: jax.Array) -> jax.Array:
    valid = ids >= 0
    safe_ids = jnp.where(valid, ids, 0)
    counts = jnp.zeros(logits.shape, dtype=jnp.int32).at[safe_ids].add(valid.astype(jnp.int32))
    return counts > 0


def apply_repetition_penalty(logits: jax.Array, recent_ids: jax.Array) -> jax.Array:
    if GEN_REPETITION_PENALTY <= 1:
        return logits

    repeated = token_mask(logits, recent_ids)
    penalized = jnp.where(logits > 0, logits / GEN_REPETITION_PENALTY, logits * GEN_REPETITION_PENALTY)
    return jnp.where(repeated, penalized, logits)


def apply_banned_tokens(logits: jax.Array, banned_ids: jax.Array) -> jax.Array:
    banned = token_mask(logits, banned_ids)
    return jnp.where(banned, -jnp.inf, logits)


def sample_next_token(logits: jax.Array, key: jax.Array) -> jax.Array:
    if GEN_TEMPERATURE <= 0:
        raise ValueError("GEN_TEMPERATURE must be > 0")

    logits = logits.astype(jnp.float32)
    logits = logits / GEN_TEMPERATURE
    logits = apply_top_k(logits, GEN_TOP_K)
    logits = apply_top_p(logits, GEN_TOP_P)
    probs = jax.nn.softmax(logits)
    return jax.random.choice(key, logits.shape[-1], p=probs)


@eqx.filter_jit
def generate_next_token(
    model: SolenaV2,
    input_ids: jax.Array,
    last_index: jax.Array,
    recent_ids: jax.Array,
    banned_ids: jax.Array,
    key: jax.Array,
) -> jax.Array:
    logits = model.logits_at(input_ids, last_index)[0]
    logits = apply_repetition_penalty(logits, recent_ids)
    logits = apply_banned_tokens(logits, banned_ids)
    return sample_next_token(logits, key)


def stop_token_ids() -> set[int]:
    sp = tokenizer.load()
    ids = {tokenizer.eos_id()}
    for piece in CHAT_BOUNDARIES:
        piece_id = sp.piece_to_id(piece)
        if piece_id != tokenizer.unk_id():
            ids.add(piece_id)
    return ids


def never_sample_token_ids() -> set[int]:
    return {
        tokenizer.pad_id(),
        tokenizer.unk_id(),
        tokenizer.bos_id(),
    }


def clean_assistant_text(ids: list[int]) -> str:
    text = tokenizer.decode(ids)
    for boundary in CHAT_BOUNDARIES:
        text = text.split(boundary, 1)[0]
    return text.strip()


def model_input(ids: list[int]) -> tuple[jax.Array, jax.Array]:
    context = ids[-SEQ_LEN:]
    last_index = len(context) - 1

    padded = [tokenizer.pad_id()] * SEQ_LEN
    padded[: len(context)] = context
    input_ids = jnp.asarray(padded, dtype=jnp.int32)[None, :]
    return input_ids, jnp.asarray(last_index, dtype=jnp.int32)


def padded_token_array(ids: list[int], size: int) -> jax.Array:
    if size <= 0:
        return jnp.asarray([-1], dtype=jnp.int32)

    padded = [-1] * size
    clipped = ids[-size:]
    padded[: len(clipped)] = clipped
    return jnp.asarray(padded, dtype=jnp.int32)


def banned_ngram_tokens(ids: list[int]) -> list[int]:
    n = GEN_NO_REPEAT_NGRAM_SIZE
    if n <= 1 or len(ids) < n - 1:
        return []

    prefix = tuple(ids[-(n - 1) :])
    banned = []
    for idx in range(len(ids) - n + 1):
        if tuple(ids[idx : idx + n - 1]) == prefix:
            banned.append(ids[idx + n - 1])
    return banned[-GEN_MAX_BANNED_TOKENS:]


def banned_generation_tokens(generated_ids: list[int], stops: set[int]) -> list[int]:
    banned = banned_ngram_tokens(generated_ids)
    banned.extend(never_sample_token_ids())
    if len(generated_ids) < GEN_MIN_NEW_TOKENS:
        banned.extend(stops)
    return banned[-GEN_MAX_BANNED_TOKENS:]


def should_stop_after_text(generated_ids: list[int]) -> bool:
    if not GEN_STOP_AFTER_SENTENCE or len(generated_ids) < GEN_MIN_NEW_TOKENS:
        return False

    text = clean_assistant_text(generated_ids).rstrip()
    return text.endswith((".", "!", "?", '."', '!"', '?"'))


def generate(model: SolenaV2, prompt: str, key: jax.Array) -> str:
    generated_ids = list(generate_ids(model, prompt, key))
    if GEN_SHOW_FULL_TEXT:
        prompt_ids = tokenizer.encode(format_prompt(prompt))
        return tokenizer.decode(prompt_ids + generated_ids)
    return clean_assistant_text(generated_ids)


def generate_ids(model: SolenaV2, prompt: str, key: jax.Array):
    prompt_text = format_prompt(prompt)
    prompt_ids = tokenizer.encode(prompt_text)
    all_ids = list(prompt_ids)
    generated_ids: list[int] = []
    stops = stop_token_ids()

    for _ in range(GEN_MAX_NEW_TOKENS):
        key, sample_key = jax.random.split(key)
        input_ids, last_index = model_input(all_ids)
        recent_ids = padded_token_array(generated_ids, GEN_REPETITION_WINDOW)
        banned_ids = padded_token_array(banned_generation_tokens(generated_ids, stops), GEN_MAX_BANNED_TOKENS)
        next_token = int(generate_next_token(model, input_ids, last_index, recent_ids, banned_ids, sample_key))

        if next_token in stops and len(generated_ids) >= GEN_MIN_NEW_TOKENS:
            break

        all_ids.append(next_token)
        generated_ids.append(next_token)
        yield next_token

        if should_stop_after_text(generated_ids):
            break


def stream_generate(model: SolenaV2, prompt: str, key: jax.Array) -> None:
    generated_ids: list[int] = []
    previous_text = ""

    print("Generating...", file=sys.stderr, flush=True)
    if GEN_SHOW_FULL_TEXT and GEN_PROMPT_MODE.lower() == "chat":
        print(format_prompt(prompt), end="", flush=True)
    elif GEN_SHOW_FULL_TEXT:
        print(prompt, end="", flush=True)

    for token_id in generate_ids(model, prompt, key):
        generated_ids.append(token_id)
        text = clean_assistant_text(generated_ids)
        chunk = text[len(previous_text) :]
        if chunk:
            print(chunk, end="", flush=True)
            previous_text = text

    print()


def main() -> None:
    args = parse_args()
    model = load_model()
    key = jax.random.PRNGKey(GEN_SEED)

    if args.prompt is not None:
        stream_generate(model, args.prompt, key)
        return

    exit_commands = {command.lower() for command in GEN_EXIT_COMMANDS}

    while True:
        prompt = input("Enter a prompt: ").strip()
        if prompt.lower() in exit_commands:
            break
        if not prompt:
            continue

        key, prompt_key = jax.random.split(key)
        stream_generate(model, prompt, prompt_key)


if __name__ == "__main__":
    main()
