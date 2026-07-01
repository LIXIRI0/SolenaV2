import json
import math
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from collections.abc import Callable


def configure_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(line_buffering=True, write_through=True)
        except (OSError, ValueError):
            pass


configure_output_streams()

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.experimental import multihost_utils

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EMBED_DIM,
    EPOCHS_PER_RUN,
    FF_DIM,
    DATASET_SEED,
    LOAD_CHECKPOINT_PATH,
    LOGIT_CHUNK_SIZE,
    LR,
    MAX_BATCHES,
    N_HEADS,
    N_LAYERS,
    NUM_DEVICES,
    OPTIMIZER,
    PARAM_DTYPE,
    PER_DEVICE_BATCH_SIZE,
    PROFILE,
    RESUME,
    SAVE_BEST_ONLY,
    SEQ_LEN,
    TRAIN_STAGE,
    TRAIN_LOSS_SYNC_INTERVAL,
    TRAIN_MASK_PATH,
    TRAIN_PREFETCH_BATCHES,
    TRAIN_TOKENS_PATH,
    USE_DATA_PARALLEL,
    USE_LOSS_MASK,
    USE_MESH,
    USE_REMAT,
    VAL_BATCHES,
    VAL_MASK_PATH,
    VAL_RATIO,
    VAL_TOKENS_PATH,
    VOCAB_SIZE,
    WEIGHT_DECAY,
)
from models.SolenaV2 import SolenaV2
from utils.dataset import load_dataset
from utils.distributed import (
    batch_sharding,
    create_data_mesh,
    global_batch_shape,
    initialize_distributed,
    is_primary_process,
    local_batch_size,
    make_global_batch,
    print_once,
    process_info,
    replicated_sharding,
)
from utils.gcs_cache import sync_checkpoint_to_gcs, sync_training_artifacts_from_gcs
from utils.gcs_cache import sync_training_logs_to_gcs
from utils import tokenizer

Batch = tuple[np.ndarray, np.ndarray, np.ndarray]
SHUTDOWN_REQUESTED = False
SHUTDOWN_SIGNAL = ""


def _handle_shutdown_signal(signum, _frame) -> None:
    global SHUTDOWN_REQUESTED, SHUTDOWN_SIGNAL
    SHUTDOWN_REQUESTED = True
    try:
        SHUTDOWN_SIGNAL = signal.Signals(signum).name
    except ValueError:
        SHUTDOWN_SIGNAL = str(signum)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)


def shutdown_requested() -> bool:
    return SHUTDOWN_REQUESTED


class BatchPrefetcher:
    def __init__(self, load_fn: Callable[[], Batch], max_prefetch: int) -> None:
        self.load_fn = load_fn
        self.queue: queue.Queue[Batch | BaseException] = queue.Queue(maxsize=max_prefetch)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                batch = self.load_fn()
            except BaseException as exc:
                self.queue.put(exc)
                return

            while not self.stop_event.is_set():
                try:
                    self.queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def next(self) -> Batch:
        item = self.queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)


def flush_loss_buffer(loss_buffer: list[jax.Array]) -> tuple[float, int]:
    if not loss_buffer:
        return 0.0, 0

    losses = jax.device_get(loss_buffer)
    loss_buffer.clear()
    return sum(float(np.asarray(loss)) for loss in losses), len(losses)


def attention_matrix_mb() -> float:
    dtype_bytes = 2 if PARAM_DTYPE == "bfloat16" else 4
    return N_HEADS * SEQ_LEN * SEQ_LEN * dtype_bytes / (1024 * 1024)


def logit_chunk_mb() -> float:
    chunk_size = min(LOGIT_CHUNK_SIZE, SEQ_LEN)
    return PER_DEVICE_BATCH_SIZE * chunk_size * VOCAB_SIZE * 4 / (1024 * 1024)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def throughput_summary(elapsed_seconds: float, batches: int) -> str:
    tokens = batches * BATCH_SIZE * SEQ_LEN
    seconds = max(elapsed_seconds, 1e-9)
    tokens_per_second = tokens / seconds
    tokens_per_minute = tokens_per_second * 60
    return (
        f"time={format_duration(elapsed_seconds)} | "
        f"tokens={tokens / 1_000_000:.1f}M | "
        f"tok/s={tokens_per_second / 1_000_000:.2f}M | "
        f"tok/min={tokens_per_minute / 1_000_000:.1f}M"
    )


def cross_entropy_loss(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    mask = mask.astype(loss.dtype)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def chunked_hidden_cross_entropy_loss(
    model: SolenaV2,
    hidden: jax.Array,
    targets: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    seq_len = hidden.shape[1]
    chunk_size = min(LOGIT_CHUNK_SIZE, seq_len)
    if seq_len % chunk_size != 0:
        pad = chunk_size - (seq_len % chunk_size)
        hidden = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0)))
        targets = jnp.pad(targets, ((0, 0), (0, pad)))
        mask = jnp.pad(mask, ((0, 0), (0, pad)))

    num_chunks = hidden.shape[1] // chunk_size
    hidden_chunks = hidden.reshape(hidden.shape[0], num_chunks, chunk_size, hidden.shape[2])
    target_chunks = targets.reshape(targets.shape[0], num_chunks, chunk_size)
    mask_chunks = mask.reshape(mask.shape[0], num_chunks, chunk_size)

    hidden_chunks = jnp.swapaxes(hidden_chunks, 0, 1)
    target_chunks = jnp.swapaxes(target_chunks, 0, 1)
    mask_chunks = jnp.swapaxes(mask_chunks, 0, 1)

    def chunk_loss(carry, xs):
        total_loss, total_weight = carry
        hidden_chunk, target_chunk, mask_chunk = xs
        logits = (hidden_chunk @ model.token_embedding.T).astype(jnp.float32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, target_chunk)
        weights = mask_chunk.astype(loss.dtype)
        total_loss = total_loss + jnp.sum(loss * weights)
        total_weight = total_weight + jnp.sum(weights)
        return (total_loss, total_weight), None

    (total_loss, total_weight), _ = jax.lax.scan(
        chunk_loss,
        (jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(0.0, dtype=jnp.float32)),
        (hidden_chunks, target_chunks, mask_chunks),
    )

    return total_loss / jnp.maximum(total_weight, 1.0)


def build_optimizer() -> optax.GradientTransformation:
    if OPTIMIZER == "adamw":
        return optax.adamw(LR, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER == "adafactor":
        return optax.adafactor(
            learning_rate=LR,
            multiply_by_parameter_scale=False,
            clipping_threshold=1.0,
            momentum=None,
            weight_decay_rate=WEIGHT_DECAY,
            factored=True,
        )
    raise ValueError(f"unknown OPTIMIZER: {OPTIMIZER}")


@eqx.filter_value_and_grad
def loss_fn(model: SolenaV2, x: jax.Array, y: jax.Array, mask: jax.Array, key: jax.Array) -> jax.Array:
    hidden = model.hidden_states(x, key=key, train=True)
    return chunked_hidden_cross_entropy_loss(model, hidden, y, mask)


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
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
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
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def eval_step(model: SolenaV2, x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    hidden = model.hidden_states(x, train=False)
    return chunked_hidden_cross_entropy_loss(model, hidden, y, mask)


@eqx.filter_pmap(axis_name="devices")
def parallel_eval_step(model: SolenaV2, x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    hidden = model.hidden_states(x, train=False)
    loss = chunked_hidden_cross_entropy_loss(model, hidden, y, mask)
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


def get_mesh_train_batch(dataset, sharding):
    local_size = local_batch_size()
    return mesh_batch_from_local(dataset.get_batch("train", batch_size=local_size), sharding)


def mesh_batch_from_local(local_batch: Batch, sharding):
    x, y, mask = local_batch
    return (
        make_global_batch(x, sharding),
        make_global_batch(y, sharding),
        make_global_batch(mask, sharding),
    )


def get_mesh_val_batch(dataset, batch_idx: int, batches: int, sharding):
    local_size = local_batch_size()
    x, y, mask = dataset.get_eval_batch_shard(
        "val",
        batch_idx,
        batches,
        global_batch_size=global_batch_shape()[0],
        local_batch_size=local_size,
        process_index=jax.process_index(),
    )
    return (
        make_global_batch(x, sharding),
        make_global_batch(y, sharding),
        make_global_batch(mask, sharding),
    )


def estimate_mesh_val_loss(model: SolenaV2, dataset, batches: int, sharding) -> float | None:
    if batches <= 0:
        return None

    losses = []
    for batch_idx in range(batches):
        vx, vy, vmask = get_mesh_val_batch(dataset, batch_idx, batches, sharding)
        loss = eval_step(model, vx, vy, vmask)
        losses.append(float(jax.device_get(loss)))

    return sum(losses) / len(losses)


def replicate_tree(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x,
        tree,
    )


def unreplicate_tree(tree):
    return jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, tree)


def checkpoint_metadata_path(path: str = CHECKPOINT_PATH) -> str:
    return f"{path}.json"


def load_checkpoint_metadata(path: str = CHECKPOINT_PATH) -> dict:
    metadata_path = checkpoint_metadata_path(path)
    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint_metadata(metadata: dict, path: str = CHECKPOINT_PATH) -> None:
    metadata_path = checkpoint_metadata_path(path)
    temp_path = f"{metadata_path}.tmp-{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(temp_path, metadata_path)


def current_process_index() -> int | None:
    if USE_MESH:
        return jax.process_index()
    return None


def flush_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass


def sync_logs_to_gcs(reason: str) -> None:
    flush_output_streams()
    sync_training_logs_to_gcs(
        process_index=current_process_index(),
        reason=reason,
        print_fn=lambda message: print(message, flush=True),
    )


def sync_last_completed_checkpoint_to_gcs() -> None:
    if USE_MESH and not is_primary_process():
        return

    if not os.path.exists(CHECKPOINT_PATH):
        print_once("shutdown requested, but no completed local checkpoint exists to upload")
        return

    print_once("shutdown requested; uploading last completed checkpoint to GCS")
    sync_checkpoint_to_gcs(print_once)


def sync_shutdown_artifacts_to_gcs() -> None:
    sync_logs_to_gcs("shutdown")
    sync_last_completed_checkpoint_to_gcs()


def model_signature() -> dict:
    return {
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "embed_dim": EMBED_DIM,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "ff_dim": FF_DIM,
        "param_dtype": PARAM_DTYPE,
    }


def file_fingerprint(path: str) -> dict:
    stat = os.stat(path)
    return {
        "path": os.path.basename(path),
        "size": stat.st_size,
    }


def dataset_fingerprint() -> dict:
    paths = [TRAIN_TOKENS_PATH, VAL_TOKENS_PATH]
    if USE_LOSS_MASK:
        if TRAIN_MASK_PATH is None or VAL_MASK_PATH is None:
            raise ValueError("mask paths must be set when USE_LOSS_MASK=True")
        paths.extend([TRAIN_MASK_PATH, VAL_MASK_PATH])

    return {
        "val_ratio": VAL_RATIO,
        "files": [file_fingerprint(path) for path in paths],
    }


def make_checkpoint_metadata(
    *,
    epoch: int,
    metric_name: str,
    best_score: float,
    train_loss: float,
    val_loss: float | None,
    current_dataset_fingerprint: dict,
    checkpoint_reason: str,
    batches_per_epoch: int | None = None,
) -> dict:
    metadata = {
        "epoch": epoch,
        "metric_name": metric_name,
        "best_score": best_score,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_signature": model_signature(),
        "dataset_fingerprint": current_dataset_fingerprint,
        "profile": PROFILE,
        "train_stage": TRAIN_STAGE,
        "optimizer": OPTIMIZER,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
        "num_devices": NUM_DEVICES,
        "checkpoint_reason": checkpoint_reason,
    }
    if batches_per_epoch is not None:
        metadata["batches_per_epoch"] = batches_per_epoch
    if SHUTDOWN_SIGNAL:
        metadata["shutdown_signal"] = SHUTDOWN_SIGNAL
    return metadata


def save_checkpoint(model: SolenaV2, metadata: dict | None = None) -> None:
    if USE_MESH:
        multihost_utils.sync_global_devices("solena_checkpoint_start")

    try:
        if shutdown_requested():
            sync_shutdown_artifacts_to_gcs()
            return
        if USE_MESH and not is_primary_process():
            return

        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        temp_path = f"{CHECKPOINT_PATH}.tmp-{os.getpid()}"
        model = jax.device_get(model)
        eqx.tree_serialise_leaves(temp_path, model)
        if shutdown_requested():
            Path(temp_path).unlink(missing_ok=True)
            sync_shutdown_artifacts_to_gcs()
            return
        os.replace(temp_path, CHECKPOINT_PATH)
        if metadata is not None:
            metadata = {
                "saved_at": time.time(),
                **metadata,
            }
            save_checkpoint_metadata(metadata)
        sync_checkpoint_to_gcs(print_once)
    finally:
        if USE_MESH:
            multihost_utils.sync_global_devices("solena_checkpoint_done")


def load_checkpoint(model: SolenaV2) -> tuple[SolenaV2, dict, bool]:
    if USE_MESH:
        multihost_utils.assert_equal(
            os.path.exists(LOAD_CHECKPOINT_PATH),
            "checkpoint existence differs across hosts; use shared storage or copy checkpoint to every VM",
        )

    if RESUME and os.path.exists(LOAD_CHECKPOINT_PATH):
        metadata = load_checkpoint_metadata(LOAD_CHECKPOINT_PATH) if LOAD_CHECKPOINT_PATH == CHECKPOINT_PATH else {}
        saved_signature = metadata.get("model_signature")
        if saved_signature is not None and saved_signature != model_signature():
            raise RuntimeError(
                "checkpoint model config does not match current config; "
                f"checkpoint={saved_signature}, current={model_signature()}. "
                "Use the matching config or delete/rename the checkpoint."
            )

        print_once(f"Loading checkpoint from {LOAD_CHECKPOINT_PATH}")
        print_once(f"Saving checkpoints to {CHECKPOINT_PATH}")
        try:
            loaded = eqx.tree_deserialise_leaves(LOAD_CHECKPOINT_PATH, model)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load checkpoint {LOAD_CHECKPOINT_PATH}; "
                "the checkpoint probably does not match the active model config"
            ) from exc
        return loaded, metadata, True
    if TRAIN_STAGE == "sft":
        raise FileNotFoundError(f"SFT needs a base or SFT checkpoint to load: {LOAD_CHECKPOINT_PATH}")
    return model, {}, False


def main() -> None:
    install_signal_handlers()
    initialize_distributed()
    sync_training_artifacts_from_gcs()
    if USE_MESH:
        multihost_utils.sync_global_devices("solena_gcs_cache_ready")

    if tokenizer.vocab_size() != VOCAB_SIZE:
        raise ValueError(
            f"tokenizer vocab size {tokenizer.vocab_size()} does not match config VOCAB_SIZE {VOCAB_SIZE}; "
            "rerun training/train_bpe.py and training/encodedata.py"
        )

    print_once(
        f"profile={PROFILE} | stage={TRAIN_STAGE} | seq_len={SEQ_LEN} | batch={BATCH_SIZE} "
        f"({NUM_DEVICES}x{PER_DEVICE_BATCH_SIZE}) | dim={EMBED_DIM} | heads={N_HEADS} | "
        f"layers={N_LAYERS} | ff={FF_DIM} | lr={LR:g} | optimizer={OPTIMIZER} | wd={WEIGHT_DECAY:g} | "
        f"dtype={PARAM_DTYPE} | remat={USE_REMAT} | "
        f"logit_chunk={LOGIT_CHUNK_SIZE} | logit_chunk_per_chip={logit_chunk_mb():.0f}MB | "
        f"attn_matrix={attention_matrix_mb():.1f}MB"
    )
    if USE_MESH:
        print_once(f"mesh mode | {process_info()}")

    dataset = load_dataset()
    if USE_MESH:
        dataset.rng = np.random.default_rng(DATASET_SEED + jax.process_index())

    current_dataset_fingerprint = dataset_fingerprint()
    total_tokens = len(dataset.train) + len(dataset.val)
    print_once(
        f"dataset | train_tokens={len(dataset.train)} | val_tokens={len(dataset.val)} "
        f"| val_share={len(dataset.val) / total_tokens:.2%}"
    )

    key = jax.random.PRNGKey(0)
    model_key, train_key = jax.random.split(key)

    model, checkpoint_metadata, resumed_from_checkpoint = load_checkpoint(SolenaV2(model_key))
    optimizer = build_optimizer()
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    best_checkpoint_score = float(checkpoint_metadata.get("best_score", float("inf")))
    best_checkpoint_metric_name = checkpoint_metadata.get("metric_name", "val_loss")
    saved_dataset_fingerprint = checkpoint_metadata.get("dataset_fingerprint")
    dataset_changed = (
        resumed_from_checkpoint
        and saved_dataset_fingerprint != current_dataset_fingerprint
    )
    if dataset_changed:
        print_once("Dataset fingerprint changed since checkpoint metadata; reseeding best checkpoint score.")
        best_checkpoint_score = float("inf")

    needs_best_score_seed = resumed_from_checkpoint and SAVE_BEST_ONLY and not math.isfinite(best_checkpoint_score)

    mesh = None
    mesh_batch_sharding = None
    if USE_MESH:
        available_devices = jax.device_count()
        if available_devices != NUM_DEVICES:
            raise RuntimeError(f"requested {NUM_DEVICES} mesh devices but JAX sees {available_devices}")

        mesh = create_data_mesh()
        mesh_batch_sharding = batch_sharding(mesh)
        model = eqx.filter_shard(model, replicated_sharding(mesh))
        opt_state = eqx.filter_shard(opt_state, replicated_sharding(mesh))
        print_once(
            f"Using mesh data parallel training over {NUM_DEVICES} global devices "
            f"with local batch {local_batch_size()} and global batch {BATCH_SIZE}"
        )

    elif USE_DATA_PARALLEL:
        available_devices = jax.local_device_count()
        if available_devices < NUM_DEVICES:
            raise RuntimeError(f"requested {NUM_DEVICES} devices but JAX only sees {available_devices}")

        print_once(
            f"Using data parallel training on {NUM_DEVICES} devices "
            f"with per-device batch {PER_DEVICE_BATCH_SIZE}"
        )
        model = replicate_tree(model)
        opt_state = replicate_tree(opt_state)

    if needs_best_score_seed:
        if USE_MESH:
            seeded_val_loss = estimate_mesh_val_loss(model, dataset, VAL_BATCHES, mesh_batch_sharding)
        elif USE_DATA_PARALLEL:
            seeded_val_loss = estimate_parallel_val_loss(model, dataset, VAL_BATCHES)
        else:
            seeded_val_loss = estimate_val_loss(model, dataset, VAL_BATCHES)

        if seeded_val_loss is not None:
            best_checkpoint_score = seeded_val_loss
            best_checkpoint_metric_name = "val_loss"
            print_once(f"Seeded best_val_loss from loaded checkpoint: {best_checkpoint_score:.4f}")

    batches_per_epoch = MAX_BATCHES
    if batches_per_epoch is None:
        batches_per_epoch = max(1, len(dataset.train) // (BATCH_SIZE * SEQ_LEN))
    print_once(f"batches_per_epoch={batches_per_epoch}")

    train_prefetcher = None
    if TRAIN_PREFETCH_BATCHES > 0:
        if USE_MESH:
            load_train_batch = lambda: dataset.get_batch("train", batch_size=local_batch_size())
        elif USE_DATA_PARALLEL:
            load_train_batch = dataset.get_sharded_train_batch
        else:
            load_train_batch = dataset.get_train_batch

        train_prefetcher = BatchPrefetcher(load_train_batch, TRAIN_PREFETCH_BATCHES)
        print_once(
            f"prefetch={TRAIN_PREFETCH_BATCHES} | "
            f"loss_sync_interval={TRAIN_LOSS_SYNC_INTERVAL}"
        )

    try:
        for epoch in range(1, EPOCHS_PER_RUN + 1):
            epoch_start_time = time.monotonic()
            running_loss = 0.0
            loss_count = 0
            last_batch_idx = 0
            loss_buffer: list[jax.Array] = []

            for batch_idx in range(1, batches_per_epoch + 1):
                last_batch_idx = batch_idx
                train_key, step_key = jax.random.split(train_key)
                if USE_MESH:
                    if train_prefetcher is None:
                        x, y, mask = get_mesh_train_batch(dataset, mesh_batch_sharding)
                    else:
                        x, y, mask = mesh_batch_from_local(train_prefetcher.next(), mesh_batch_sharding)
                    step_key = jax.device_put(step_key, replicated_sharding(mesh))
                    model, opt_state, loss = train_step(
                        model,
                        opt_state,
                        optimizer,
                        x,
                        y,
                        mask,
                        step_key,
                    )
                    loss_buffer.append(loss)
                elif USE_DATA_PARALLEL:
                    if train_prefetcher is None:
                        x, y, mask = dataset.get_sharded_train_batch()
                    else:
                        x, y, mask = train_prefetcher.next()
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
                    loss_buffer.append(loss[0])
                else:
                    if train_prefetcher is None:
                        x, y, mask = dataset.get_train_batch()
                    else:
                        x, y, mask = train_prefetcher.next()
                    model, opt_state, loss = train_step(
                        model,
                        opt_state,
                        optimizer,
                        jnp.asarray(x),
                        jnp.asarray(y),
                        jnp.asarray(mask),
                        step_key,
                    )
                    loss_buffer.append(loss)

                if batch_idx % TRAIN_LOSS_SYNC_INTERVAL == 0:
                    loss_sum, sample_count = flush_loss_buffer(loss_buffer)
                    running_loss += loss_sum
                    loss_count += sample_count

                if shutdown_requested():
                    print_once(
                        f"shutdown requested by {SHUTDOWN_SIGNAL or 'signal'}; "
                        f"abandoning epoch {epoch} at batch {last_batch_idx}/{batches_per_epoch}"
                    )
                    break

            if shutdown_requested():
                sync_shutdown_artifacts_to_gcs()
                return

            loss_sum, sample_count = flush_loss_buffer(loss_buffer)
            running_loss += loss_sum
            loss_count += sample_count
            train_loss = running_loss / max(loss_count, 1)

            if shutdown_requested():
                sync_shutdown_artifacts_to_gcs()
                return

            if USE_MESH:
                val_loss = estimate_mesh_val_loss(model, dataset, VAL_BATCHES, mesh_batch_sharding)
                checkpoint_model = model
            elif USE_DATA_PARALLEL:
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
                    make_checkpoint_metadata(
                        epoch=epoch,
                        metric_name=checkpoint_metric_name,
                        best_score=best_checkpoint_score,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        current_dataset_fingerprint=current_dataset_fingerprint,
                        checkpoint_reason="epoch",
                        batches_per_epoch=batches_per_epoch,
                    ),
                )
                checkpoint_status = "saved"
            else:
                checkpoint_status = "skipped"

            epoch_elapsed = time.monotonic() - epoch_start_time
            epoch_throughput = throughput_summary(epoch_elapsed, last_batch_idx)

            if val_loss is None:
                print_once(
                    f"epoch {epoch}/{EPOCHS_PER_RUN} | train_loss={train_loss:.4f} | "
                    f"checkpoint={checkpoint_status} | best_{best_checkpoint_metric_name}={best_checkpoint_score:.4f} | "
                    f"{epoch_throughput}"
                )
            else:
                print_once(
                    f"epoch {epoch}/{EPOCHS_PER_RUN} | train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | checkpoint={checkpoint_status} | "
                    f"best_{best_checkpoint_metric_name}={best_checkpoint_score:.4f} | "
                    f"{epoch_throughput}"
                )
            sync_logs_to_gcs(f"epoch-{epoch}")
    finally:
        if train_prefetcher is not None:
            train_prefetcher.stop()
if __name__ == "__main__":
    main()
