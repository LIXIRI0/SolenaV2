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
    EPOCHS_PER_RUN,
    LR,
    MAX_BATCHES,
    RESUME,
    SAVE_BEST_ONLY,
    SEQ_LEN,
    VAL_BATCHES,
)
from models.SolenaV2 import SolenaV2
from utils.dataset import load_dataset


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return jnp.mean(loss)


@eqx.filter_value_and_grad
def loss_fn(model: SolenaV2, x: jax.Array, y: jax.Array, key: jax.Array) -> jax.Array:
    logits = model(x, key=key, train=True)
    return cross_entropy_loss(logits, y)


@eqx.filter_jit
def train_step(
    model: SolenaV2,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
) -> tuple[SolenaV2, optax.OptState, jax.Array]:
    loss, grads = loss_fn(model, x, y, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model: SolenaV2, x: jax.Array, y: jax.Array) -> jax.Array:
    logits = model(x, train=False)
    return cross_entropy_loss(logits, y)


def estimate_val_loss(model: SolenaV2, dataset, batches: int) -> float | None:
    if batches <= 0:
        return None

    losses = []
    for _ in range(batches):
        vx, vy = dataset.get_val_batch()
        loss = eval_step(model, jnp.asarray(vx), jnp.asarray(vy))
        losses.append(float(loss))

    return sum(losses) / len(losses)


def save_checkpoint(model: SolenaV2) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    eqx.tree_serialise_leaves(CHECKPOINT_PATH, model)


def load_checkpoint(model: SolenaV2) -> SolenaV2:
    if RESUME and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        return eqx.tree_deserialise_leaves(CHECKPOINT_PATH, model)
    return model


def main() -> None:
    dataset = load_dataset()
    key = jax.random.PRNGKey(0)
    model_key, train_key = jax.random.split(key)

    model = load_checkpoint(SolenaV2(model_key))
    optimizer = optax.adamw(LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    batches_per_epoch = MAX_BATCHES
    if batches_per_epoch is None:
        batches_per_epoch = max(1, len(dataset.train) // (BATCH_SIZE * SEQ_LEN))

    best_checkpoint_score = float("inf")

    for epoch in range(1, EPOCHS_PER_RUN + 1):
        running_loss = 0.0

        for batch_idx in range(1, batches_per_epoch + 1):
            train_key, step_key = jax.random.split(train_key)
            x, y = dataset.get_train_batch()
            model, opt_state, loss = train_step(
                model,
                opt_state,
                optimizer,
                jnp.asarray(x),
                jnp.asarray(y),
                step_key,
            )
            running_loss += float(loss)

        train_loss = running_loss / batches_per_epoch
        val_loss = estimate_val_loss(model, dataset, VAL_BATCHES)

        checkpoint_metric_name = "val_loss" if val_loss is not None else "train_loss"
        checkpoint_score = val_loss if val_loss is not None else train_loss
        is_best_epoch = checkpoint_score < best_checkpoint_score

        if not SAVE_BEST_ONLY or is_best_epoch:
            best_checkpoint_score = min(best_checkpoint_score, checkpoint_score)
            save_checkpoint(model)
            checkpoint_status = "saved"
        else:
            checkpoint_status = "skipped"

        if val_loss is None:
            print(
                f"epoch {epoch}/{EPOCHS_PER_RUN} | train_loss={train_loss:.4f} | "
                f"checkpoint={checkpoint_status} | best_{checkpoint_metric_name}={best_checkpoint_score:.4f}"
            )
        else:
            print(
                f"epoch {epoch}/{EPOCHS_PER_RUN} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | checkpoint={checkpoint_status} | "
                f"best_{checkpoint_metric_name}={best_checkpoint_score:.4f}"
            )
if __name__ == "__main__":
    main()
