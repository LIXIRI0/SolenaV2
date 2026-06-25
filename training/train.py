import json
import math
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EMBED_DIM,
    EPOCHS_PER_RUN,
    FF_DIM,
    LOAD_CHECKPOINT_PATH,
    LR,
    MAX_BATCHES,
    N_HEADS,
    N_LAYERS,
    NUM_DEVICES,
    PER_DEVICE_BATCH_SIZE,
    PROFILE,
    RESUME,
    SAVE_BEST_ONLY,
    SEQ_LEN,
    TRAIN_STAGE,
    USE_DATA_PARALLEL,
    USE_REMAT,
    VAL_BATCHES,
    VOCAB_SIZE,
)
from models.SolenaV2 import SolenaV2
from utils.dataset import load_dataset
from utils import tokenizer


def cross_entropy_loss(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    mask = mask.astype(loss.dtype)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@eqx.filter_value_and_grad
def loss_fn(model: SolenaV2, x: jax.Array, y: jax.Array, mask: jax.Array, key: jax.Array) -> jax.Array:
    logits = model(x, key=key, train=True)
    return cross_entropy_loss(logits, y, mask)


@eqx.filter_jit
def train_step(
    model: SolenaV2,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    key: jax.Array,
) -> tuple[SolenaV2, optax.OptState, jax.Array]:
    loss, grads = loss_fn(model, x, y, mask, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_pmap(axis_name="devices")
def parallel_train_step(
    model: SolenaV2,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x: jax.Array,
    y: jax.Array,
    mask: jax.Array,
    key: jax.Array,
) -> tuple[SolenaV2, optax.OptState, jax.Array]:
    loss, grads = loss_fn(model, x, y, mask, key)
    loss = jax.lax.pmean(loss, axis_name="devices")
    grads = jax.lax.pmean(grads, axis_name="devices")
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model: SolenaV2, x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    logits = model(x, train=False)
    return cross_entropy_loss(logits, y, mask)


@eqx.filter_pmap(axis_name="devices")
def parallel_eval_step(model: SolenaV2, x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    logits = model(x, train=False)
    loss = cross_entropy_loss(logits, y, mask)
    return jax.lax.pmean(loss, axis_name="devices")


def estimate_val_loss(model: SolenaV2, dataset, batches: int) -> float | None:
    if batches <= 0:
        return None

    losses = []
    for batch_idx in range(batches):
        vx, vy, vmask = dataset.get_val_eval_batch(batch_idx, batches)
        loss = eval_step(model, jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(vmask))
        losses.append(float(loss))

    return sum(losses) / len(losses)


def estimate_parallel_val_loss(model: SolenaV2, dataset, batches: int) -> float | None:
    if batches <= 0:
        return None

    losses = []
    for batch_idx in range(batches):
        vx, vy, vmask = dataset.get_sharded_val_eval_batch(batch_idx, batches)
        loss = parallel_eval_step(model, jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(vmask))
        losses.append(float(jax.device_get(loss[0])))

    return sum(losses) / len(losses)


def replicate_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x,
        tree,
    )


def unreplicate_tree(tree):
    return jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, tree)


def checkpoint_metadata_path() -> str:
    return f"{CHECKPOINT_PATH}.json"


def load_checkpoint_metadata() -> dict:
    path = checkpoint_metadata_path()
    if not os.path.exists(path):
        return {}

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint_metadata(metadata: dict) -> None:
    with open(checkpoint_metadata_path(), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_checkpoint(model: SolenaV2, metadata: dict | None = None) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    eqx.tree_serialise_leaves(CHECKPOINT_PATH, model)
    if metadata is not None:
        save_checkpoint_metadata(metadata)


def load_checkpoint(model: SolenaV2) -> tuple[SolenaV2, dict, bool]:
    if RESUME and os.path.exists(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint from {LOAD_CHECKPOINT_PATH}")
        print(f"Saving checkpoints to {CHECKPOINT_PATH}")
        return eqx.tree_deserialise_leaves(LOAD_CHECKPOINT_PATH, model), load_checkpoint_metadata(), True
    if TRAIN_STAGE == "sft":
        raise FileNotFoundError(f"SFT needs a base or SFT checkpoint to load: {LOAD_CHECKPOINT_PATH}")
    return model, {}, False


def main() -> None:
    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size()} does not match config VOCAB_SIZE {VOCAB_SIZE}; "
            "rerun training/train_bpe.py and training/encodedata.py"
        )

    print(
        f"profile={PROFILE} | stage={TRAIN_STAGE} | seq_len={SEQ_LEN} | batch={BATCH_SIZE} "
        f"({NUM_DEVICES}x{PER_DEVICE_BATCH_SIZE}) | dim={EMBED_DIM} | heads={N_HEADS} | "
        f"layers={N_LAYERS} | ff={FF_DIM} | lr={LR:g} | remat={USE_REMAT}"
    )

    dataset = load_dataset()
    key = jax.random.PRNGKey(0)
    model_key, train_key = jax.random.split(key)

    model, checkpoint_metadata, resumed_from_checkpoint = load_checkpoint(SolenaV2(model_key))
    optimizer = optax.adamw(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_checkpoint_score = float(checkpoint_metadata.get("best_score", float("inf")))
    best_checkpoint_metric_name = checkpoint_metadata.get("metric_name", "val_loss")
    needs_best_score_seed = resumed_from_checkpoint and SAVE_BEST_ONLY and not math.isfinite(best_checkpoint_score)

    if USE_DATA_PARALLEL:
        available_devices = jax.local_device_count()
        if available_devices < NUM_DEVICES:
            raise RuntimeError(f"requested {NUM_DEVICES} devices but JAX only sees {available_devices}")

        print(
            f"Using data parallel training on {NUM_DEVICES} devices "
            f"with per-device batch {PER_DEVICE_BATCH_SIZE}"
        )
        model = replicate_tree(model)
        opt_state = replicate_tree(opt_state)

    if needs_best_score_seed:
        if USE_DATA_PARALLEL:
            seeded_val_loss = estimate_parallel_val_loss(model, dataset, VAL_BATCHES)
        else:
            seeded_val_loss = estimate_val_loss(model, dataset, VAL_BATCHES)

        if seeded_val_loss is not None:
            best_checkpoint_score = seeded_val_loss
            best_checkpoint_metric_name = "val_loss"
            print(f"Seeded best_val_loss from loaded checkpoint: {best_checkpoint_score:.4f}")

    batches_per_epoch = MAX_BATCHES
    if batches_per_epoch is None:
        batches_per_epoch = max(1, len(dataset.train) // (BATCH_SIZE * SEQ_LEN))

    for epoch in range(1, EPOCHS_PER_RUN + 1):
        running_loss = 0.0

        for batch_idx in range(1, batches_per_epoch + 1):
            train_key, step_key = jax.random.split(train_key)
            if USE_DATA_PARALLEL:
                x, y, mask = dataset.get_sharded_train_batch()
                step_keys = jax.random.split(step_key, NUM_DEVICES)
                model, opt_state, loss = parallel_train_step(
                    model,
                    opt_state,
                    optimizer,
                    jnp.asarray(x),
                    jnp.asarray(y),
                    jnp.asarray(mask),
                    step_keys,
                )
                running_loss += float(jax.device_get(loss[0]))
            else:
                x, y, mask = dataset.get_train_batch()
                model, opt_state, loss = train_step(
                    model,
                    opt_state,
                    optimizer,
                    jnp.asarray(x),
                    jnp.asarray(y),
                    jnp.asarray(mask),
                    step_key,
                )
                running_loss += float(loss)

        train_loss = running_loss / batches_per_epoch
        if USE_DATA_PARALLEL:
            val_loss = estimate_parallel_val_loss(model, dataset, VAL_BATCHES)
            checkpoint_model = unreplicate_tree(model)
        else:
            val_loss = estimate_val_loss(model, dataset, VAL_BATCHES)
            checkpoint_model = model

        checkpoint_metric_name = "val_loss" if val_loss is not None else "train_loss"
        checkpoint_score = val_loss if val_loss is not None else train_loss
        is_best_epoch = checkpoint_score < best_checkpoint_score

        if not SAVE_BEST_ONLY or is_best_epoch:
            best_checkpoint_score = checkpoint_score
            best_checkpoint_metric_name = checkpoint_metric_name
            save_checkpoint(
                checkpoint_model,
                {
                    "epoch": epoch,
                    "metric_name": checkpoint_metric_name,
                    "best_score": best_checkpoint_score,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
            )
            checkpoint_status = "saved"
        else:
            checkpoint_status = "skipped"

        if val_loss is None:
            print(
                f"epoch {epoch}/{EPOCHS_PER_RUN} | train_loss={train_loss:.4f} | "
                f"checkpoint={checkpoint_status} | best_{best_checkpoint_metric_name}={best_checkpoint_score:.4f}"
            )
        else:
            print(
                f"epoch {epoch}/{EPOCHS_PER_RUN} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | checkpoint={checkpoint_status} | "
                f"best_{best_checkpoint_metric_name}={best_checkpoint_score:.4f}"
            )
if __name__ == "__main__":
    main()
