import os
import socket
import subprocess
import time
from pathlib import Path

from config import (
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    DATA_DIR,
    DATA_PATH,
    GCS_SYNC_LOGS,
    GCS_ROOT,
    GCS_SYNC_CHECKPOINTS,
    LOAD_CHECKPOINT_PATH,
    LOG_DIR,
    PRETRAIN_DATA_PATH,
    PROFILE,
    RESUME,
    TOKENIZER_PATH,
    TRAIN_LOG_PATH,
    TRAIN_STAGE,
    TRAIN_MASK_PATH,
    TRAIN_TOKENS_PATH,
    USE_LOSS_MASK,
    VAL_MASK_PATH,
    VAL_TOKENS_PATH,
)


LOCK_TIMEOUT_SECONDS = 30 * 60


def gcs_enabled() -> bool:
    return GCS_ROOT.startswith("gs://")


def validate_gcs_root() -> None:
    if "your-bucket" in GCS_ROOT:
        raise RuntimeError("replace TRC_GCS_ROOT in config.py with your real gs:// bucket prefix")


def gcs_join(*parts: str) -> str:
    if not parts:
        raise ValueError("gcs_join needs at least one part")

    root = parts[0].rstrip("/")
    rest = [part.strip("/") for part in parts[1:] if part]
    return "/".join([root, *rest])


def checkpoint_metadata_path(path: str) -> str:
    return f"{path}.json"


def remote_for_path(path: str) -> str:
    local = Path(path).resolve()
    data_dir = Path(DATA_DIR).resolve()
    checkpoint_dir = Path(CHECKPOINT_DIR).resolve()

    if local.is_relative_to(data_dir):
        return gcs_join(GCS_ROOT, "data", str(local.relative_to(data_dir)))
    if local.is_relative_to(checkpoint_dir):
        return gcs_join(GCS_ROOT, "checkpoints", str(local.relative_to(checkpoint_dir)))
    return gcs_join(GCS_ROOT, local.name)


def _storage_command() -> list[str]:
    return ["gcloud", "storage"]


def _run_storage(args: list[str], required: bool = True) -> subprocess.CompletedProcess:
    cmd = [*_storage_command(), *args]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("gcloud CLI was not found; install/auth gcloud before using SOLENA_GCS_ROOT") from exc
    if required and result.returncode != 0:
        raise RuntimeError(
            f"storage command failed: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def remote_exists(uri: str) -> bool:
    return _run_storage(["ls", uri], required=False).returncode == 0


def _acquire_lock(lock_path: Path) -> None:
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
            return
        except FileExistsError:
            try:
                lock_age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if lock_age > LOCK_TIMEOUT_SECONDS:
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() - start > LOCK_TIMEOUT_SECONDS:
                raise TimeoutError(f"timed out waiting for cache lock: {lock_path}")
            time.sleep(1)


def copy_from_gcs(remote_uri: str, local_path: str, required: bool = True) -> bool:
    local = Path(local_path)
    if local.exists() and local.stat().st_size > 0:
        return False

    if not required and not remote_exists(remote_uri):
        return False

    local.parent.mkdir(parents=True, exist_ok=True)
    lock_path = local.with_suffix(f"{local.suffix}.lock")
    temp_path = local.with_suffix(f"{local.suffix}.download-{os.getpid()}.tmp")

    _acquire_lock(lock_path)
    try:
        if local.exists() and local.stat().st_size > 0:
            return False

        temp_path.unlink(missing_ok=True)
        result = _run_storage(["cp", remote_uri, str(temp_path)], required=required)
        if result.returncode != 0:
            return False
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"downloaded empty artifact from {remote_uri}")
        temp_path.replace(local)
        return True
    finally:
        temp_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def copy_to_gcs(local_path: str, remote_uri: str) -> bool:
    if not gcs_enabled():
        return False

    validate_gcs_root()
    local = Path(local_path)
    if not local.exists() or local.stat().st_size == 0:
        return False

    _run_storage(["cp", str(local), remote_uri], required=True)
    return True


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in value)


def upload_artifacts(paths: list[str], label: str, print_fn=print) -> None:
    if not gcs_enabled():
        return

    validate_gcs_root()
    for path in paths:
        remote = remote_for_path(path)
        if copy_to_gcs(path, remote):
            print_fn(f"gcs {label} | uploaded | {path} -> {remote}")


def download_artifacts(paths: list[str], label: str, required: bool = True, print_fn=print) -> None:
    if not gcs_enabled():
        return

    validate_gcs_root()
    for path in paths:
        remote = remote_for_path(path)
        copied = copy_from_gcs(remote, path, required=required)
        status = "downloaded" if copied else "cached"
        if copied or required:
            print_fn(f"gcs {label} | {status} | {remote} -> {path}")


def raw_data_artifacts() -> list[str]:
    return [DATA_PATH]


def pretrain_tokenizer_inputs() -> list[str]:
    return [PRETRAIN_DATA_PATH]


def training_artifacts() -> tuple[list[str], list[str]]:
    required = [
        TRAIN_TOKENS_PATH,
        VAL_TOKENS_PATH,
        TOKENIZER_PATH,
    ]
    if USE_LOSS_MASK:
        if TRAIN_MASK_PATH is None or VAL_MASK_PATH is None:
            raise ValueError("mask paths must be set when USE_LOSS_MASK=True")
        required.extend([TRAIN_MASK_PATH, VAL_MASK_PATH])

    optional = []
    if RESUME:
        optional.extend(
            [
                LOAD_CHECKPOINT_PATH,
                checkpoint_metadata_path(LOAD_CHECKPOINT_PATH),
            ]
        )

    return required, optional


def encoded_artifacts() -> list[str]:
    paths = [
        TRAIN_TOKENS_PATH,
        VAL_TOKENS_PATH,
    ]
    if USE_LOSS_MASK:
        if TRAIN_MASK_PATH is None or VAL_MASK_PATH is None:
            raise ValueError("mask paths must be set when USE_LOSS_MASK=True")
        paths.extend([TRAIN_MASK_PATH, VAL_MASK_PATH])
    return paths


def tokenizer_artifacts() -> list[str]:
    paths = [TOKENIZER_PATH]
    vocab_path = str(Path(TOKENIZER_PATH).with_suffix(".vocab"))
    paths.append(vocab_path)
    return paths


def log_artifacts() -> list[str]:
    candidates: list[Path] = []
    train_log = Path(TRAIN_LOG_PATH)
    candidates.append(train_log)

    log_dir = Path(LOG_DIR)
    if log_dir.exists():
        candidates.extend(sorted(log_dir.glob("*.log")))

    paths = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen or not resolved.exists() or resolved.stat().st_size == 0:
            continue
        seen.add(resolved)
        paths.append(str(resolved))
    return paths


def sync_training_logs_to_gcs(
    process_index: int | None = None,
    reason: str = "sync",
    print_fn=print,
) -> None:
    if not gcs_enabled() or not GCS_SYNC_LOGS:
        return

    validate_gcs_root()
    host = _safe_name(socket.gethostname())
    process_name = "process-unknown" if process_index is None else f"process-{process_index}"
    reason = _safe_name(reason)

    for path in log_artifacts():
        filename = _safe_name(Path(path).name)
        remote = gcs_join(
            GCS_ROOT,
            "logs",
            TRAIN_STAGE,
            PROFILE,
            host,
            f"{process_name}-{filename}",
        )
        if copy_to_gcs(path, remote):
            print_fn(f"gcs logs | uploaded | {reason} | {path} -> {remote}")


def sync_encoded_artifacts_to_gcs(print_fn=print) -> None:
    upload_artifacts(encoded_artifacts(), f"{TRAIN_STAGE} data", print_fn)


def sync_raw_data_to_gcs(print_fn=print) -> None:
    upload_artifacts(raw_data_artifacts(), f"{TRAIN_STAGE} raw", print_fn)


def sync_pretrain_tokenizer_inputs_from_gcs(print_fn=print) -> None:
    download_artifacts(pretrain_tokenizer_inputs(), "tokenizer input", required=True, print_fn=print_fn)


def sync_encoding_inputs_from_gcs(print_fn=print) -> None:
    download_artifacts(raw_data_artifacts(), f"{TRAIN_STAGE} raw", required=True, print_fn=print_fn)
    download_artifacts([TOKENIZER_PATH], "tokenizer", required=True, print_fn=print_fn)
    download_artifacts([str(Path(TOKENIZER_PATH).with_suffix(".vocab"))], "tokenizer", required=False, print_fn=print_fn)


def sync_tokenizer_to_gcs(print_fn=print) -> None:
    upload_artifacts(tokenizer_artifacts(), "tokenizer", print_fn)


def sync_training_artifacts_from_gcs(print_fn=print) -> None:
    if not gcs_enabled():
        return

    validate_gcs_root()
    required, optional = training_artifacts()
    host = socket.gethostname()
    print_fn(f"gcs cache | host={host} | root={GCS_ROOT}")

    for path in required:
        remote = remote_for_path(path)
        copied = copy_from_gcs(remote, path, required=True)
        status = "downloaded" if copied else "cached"
        print_fn(f"gcs cache | {status} | {remote} -> {path}")

    for path in optional:
        remote = remote_for_path(path)
        copied = copy_from_gcs(remote, path, required=False)
        if copied:
            print_fn(f"gcs cache | downloaded optional | {remote} -> {path}")


def sync_checkpoint_to_gcs(print_fn=print) -> None:
    if not gcs_enabled() or not GCS_SYNC_CHECKPOINTS:
        return

    upload_artifacts([CHECKPOINT_PATH, checkpoint_metadata_path(CHECKPOINT_PATH)], "checkpoint", print_fn)
