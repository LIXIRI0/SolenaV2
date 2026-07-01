import os

PROFILE = os.getenv("SOLENA_PROFILE", "trc_tpu_16")  # "cpu_dev", "cpu_full", "collab_tpu", "kaggle_tpu_8", "trc_tpu_16", "trc_tpu_64", "tpu_train"
TRAIN_STAGE = "pretrain"     # "pretrain" or "sft"

TRC_GCS_ROOT = "gs://solena/solena"
TRC_DATA_DIR = "/tmp/solena-data"
TRC_CHECKPOINT_DIR = "/tmp/solena-checkpoints"

if PROFILE == "cpu_dev":
    VOCAB_SIZE     = 8000
    SEQ_LEN        = 64
    BATCH_SIZE     = 4
    EMBED_DIM      = 64
    N_HEADS        = 2
    N_LAYERS       = 2
    FF_DIM         = 256
    LR             = 3e-4
    MAX_BATCHES    = 500
    VAL_BATCHES    = 20
    EPOCHS_PER_RUN = 5
    MAX_EPOCHS     = 10

elif PROFILE == "cpu_full":
    VOCAB_SIZE     = 8000
    SEQ_LEN        = 128
    BATCH_SIZE     = 8
    EMBED_DIM      = 128
    N_HEADS        = 4
    N_LAYERS       = 4
    FF_DIM         = 512
    LR             = 3e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 20
    EPOCHS_PER_RUN = 50
    MAX_EPOCHS     = None

elif PROFILE == "collab_tpu":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 512
    NUM_DEVICES    = 8
    PER_DEVICE_BATCH_SIZE = 6
    BATCH_SIZE     = NUM_DEVICES * PER_DEVICE_BATCH_SIZE
    EMBED_DIM      = 512
    N_HEADS        = 8
    N_LAYERS       = 8
    FF_DIM         = 2048
    LR             = 5e-5
    MAX_BATCHES    = None
    VAL_BATCHES    = 30
    EPOCHS_PER_RUN = 10
    MAX_EPOCHS     = None

elif PROFILE == "kaggle_tpu_8":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 768
    NUM_DEVICES    = 8
    PER_DEVICE_BATCH_SIZE = 1
    BATCH_SIZE     = NUM_DEVICES * PER_DEVICE_BATCH_SIZE
    EMBED_DIM      = 1024
    N_HEADS        = 8
    N_LAYERS       = 24
    FF_DIM         = 4096
    LR             = 6e-5
    MAX_BATCHES    = 12_000
    VAL_BATCHES    = 80
    EPOCHS_PER_RUN = 20
    MAX_EPOCHS     = None
    OPTIMIZER      = "adafactor"
    PARAM_DTYPE    = "bfloat16"

elif PROFILE == "trc_tpu_64":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 768
    NUM_DEVICES    = 64
    PER_DEVICE_BATCH_SIZE = 16
    BATCH_SIZE     = NUM_DEVICES * PER_DEVICE_BATCH_SIZE
    EMBED_DIM      = 1024
    N_HEADS        = 8
    N_LAYERS       = 24
    FF_DIM         = 4096
    LR             = 1e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 120
    EPOCHS_PER_RUN = 10
    MAX_EPOCHS     = None
    OPTIMIZER      = "adafactor"
    PARAM_DTYPE    = "bfloat16"
    USE_MESH       = True

elif PROFILE == "trc_tpu_16":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 1024
    NUM_DEVICES    = 16
    PER_DEVICE_BATCH_SIZE = 128
    BATCH_SIZE     = NUM_DEVICES * PER_DEVICE_BATCH_SIZE
    EMBED_DIM      = 1024
    N_HEADS        = 8
    N_LAYERS       = 24
    FF_DIM         = 4096
    LR             = 1e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 120
    EPOCHS_PER_RUN = 60
    MAX_EPOCHS     = None
    OPTIMIZER      = "adafactor"
    PARAM_DTYPE    = "bfloat16"
    USE_MESH       = True

elif PROFILE == "tpu_train":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 768
    NUM_DEVICES    = 8
    PER_DEVICE_BATCH_SIZE = 1
    BATCH_SIZE     = NUM_DEVICES * PER_DEVICE_BATCH_SIZE
    EMBED_DIM      = 1024
    N_HEADS        = 8
    N_LAYERS       = 24
    FF_DIM         = 4096
    LR             = 1.5e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 50
    EPOCHS_PER_RUN = 100
    MAX_EPOCHS     = None
    GRAD_ACCUM_STEPS = 4
    OPTIMIZER      = "adafactor"
    PARAM_DTYPE    = "bfloat16"

else:
    raise ValueError(f"unknown PROFILE: {PROFILE}")

if "NUM_DEVICES" not in globals():
    NUM_DEVICES = 1
if "PER_DEVICE_BATCH_SIZE" not in globals():
    PER_DEVICE_BATCH_SIZE = BATCH_SIZE
if "OPTIMIZER" not in globals():
    OPTIMIZER = "adamw"
if "PARAM_DTYPE" not in globals():
    PARAM_DTYPE = "float32"
if "USE_MESH" not in globals():
    USE_MESH = False
# AdamW-style coefficient. Adafactor converts this to LR * WEIGHT_DECAY internally.
WEIGHT_DECAY = float(os.getenv("SOLENA_WEIGHT_DECAY", "0.01"))
USE_DATA_PARALLEL = NUM_DEVICES > 1
USE_REMAT = PROFILE in ("kaggle_tpu_8", "trc_tpu_16", "trc_tpu_64", "tpu_train")


def default_logit_chunk_size() -> int:
    if not PROFILE.startswith("trc_tpu_"):
        return 64

    target_mb = int(os.getenv("SOLENA_LOGIT_CHUNK_TARGET_MB", "4096"))
    bytes_per_logit = 4
    chunk_tokens = (target_mb * 1024 * 1024) // max(1, PER_DEVICE_BATCH_SIZE * VOCAB_SIZE * bytes_per_logit)
    chunk_tokens = max(64, min(SEQ_LEN, int(chunk_tokens)))
    if chunk_tokens == SEQ_LEN:
        return chunk_tokens
    return max(64, (chunk_tokens // 64) * 64)


DEFAULT_LOGIT_CHUNK_SIZE = default_logit_chunk_size()
LOGIT_CHUNK_SIZE = int(os.getenv("SOLENA_LOGIT_CHUNK_SIZE", str(DEFAULT_LOGIT_CHUNK_SIZE)))
DATASET_SEED = 1337
DISTRIBUTED_INIT_TIMEOUT = int(os.getenv("JAX_INIT_TIMEOUT", "600"))
TRAIN_PREFETCH_BATCHES = int(os.getenv("SOLENA_PREFETCH_BATCHES", "2"))
TRAIN_LOSS_SYNC_INTERVAL = int(os.getenv("SOLENA_LOSS_SYNC_INTERVAL", "32"))

if TRAIN_STAGE == "sft":
    LR = 1e-5
    EPOCHS_PER_RUN = 3
elif TRAIN_STAGE != "pretrain":
    raise ValueError(f"unknown TRAIN_STAGE: {TRAIN_STAGE}")

GEN_MAX_NEW_TOKENS = 256
GEN_MIN_NEW_TOKENS = 48
GEN_TEMPERATURE    = 0.8
GEN_TOP_P          = 0.92
GEN_TOP_K          = 50
GEN_REPETITION_PENALTY = 1.08
GEN_REPETITION_WINDOW  = 192
GEN_NO_REPEAT_NGRAM_SIZE = 5
GEN_MAX_BANNED_TOKENS = 128
GEN_STOP_AFTER_SENTENCE = False
GEN_SEED           = 0
GEN_SHOW_FULL_TEXT = False
GEN_PROMPT_MODE    = "chat" if TRAIN_STAGE == "sft" else "plain"
GEN_EXIT_COMMANDS  = ("exit", "quit", "q")

PRETRAIN_TARGET_TOKENS = 1_000_000_000
PRETRAIN_CHARS_PER_TOKEN = 4
VAL_RATIO = 0.05
PRETRAIN_SEED = 42
PRETRAIN_HF_TIMEOUT = 120
PRETRAIN_SHUFFLE_BUFFER_SIZE = 0
PRETRAIN_MIX = {
    "clean_web": 0.45,
    "wiki": 0.25,
    "textbooks": 0.15,
    "stories": 0.10,
    "cs": 0.05,
}

SFT_PERSONA_NAME = "Solena"
SFT_CREATOR_NAME = "Lixiri"
SFT_PERSONA_REPEATS = 192
SFT_PERSONA_STYLE = (
    "helpful, warm, direct, lightly playful, and honest when unsure"
)

DROPOUT = 0.1

ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = TRC_DATA_DIR if PROFILE.startswith("trc_tpu_") else os.path.join(ROOT_DIR, "data")
DEFAULT_CHECKPOINT_DIR = TRC_CHECKPOINT_DIR if PROFILE.startswith("trc_tpu_") else os.path.join(ROOT_DIR, "checkpoints")
DEFAULT_GCS_ROOT = TRC_GCS_ROOT if PROFILE.startswith("trc_tpu_") else ""

DATA_DIR         = os.getenv("SOLENA_DATA_DIR", DEFAULT_DATA_DIR)
CHECKPOINT_DIR   = os.getenv("SOLENA_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR)
LOG_DIR          = os.getenv("SOLENA_LOG_DIR", os.path.join(ROOT_DIR, "logs"))
TRAIN_LOG_PATH   = os.getenv("SOLENA_TRAIN_LOG_PATH", os.path.join(LOG_DIR, "train.log"))
GCS_ROOT         = os.getenv("SOLENA_GCS_ROOT", DEFAULT_GCS_ROOT).rstrip("/")
GCS_SYNC_CHECKPOINTS = os.getenv("SOLENA_GCS_SYNC_CHECKPOINTS", "1") != "0"
GCS_SYNC_LOGS = os.getenv("SOLENA_GCS_SYNC_LOGS", "1") != "0"

PRETRAIN_DATA_PATH = os.path.join(DATA_DIR, "raw.txt")
SFT_DATA_PATH      = os.path.join(DATA_DIR, "sft_raw.txt")

PRETRAIN_TRAIN_TOKENS_PATH = os.path.join(DATA_DIR, "train.npy")
PRETRAIN_VAL_TOKENS_PATH   = os.path.join(DATA_DIR, "val.npy")
SFT_TRAIN_TOKENS_PATH      = os.path.join(DATA_DIR, "sft_train.npy")
SFT_VAL_TOKENS_PATH        = os.path.join(DATA_DIR, "sft_val.npy")
SFT_TRAIN_MASK_PATH        = os.path.join(DATA_DIR, "sft_train_mask.npy")
SFT_VAL_MASK_PATH          = os.path.join(DATA_DIR, "sft_val_mask.npy")

DATA_PATH = SFT_DATA_PATH if TRAIN_STAGE == "sft" else PRETRAIN_DATA_PATH
TRAIN_TOKENS_PATH = SFT_TRAIN_TOKENS_PATH if TRAIN_STAGE == "sft" else PRETRAIN_TRAIN_TOKENS_PATH
VAL_TOKENS_PATH = SFT_VAL_TOKENS_PATH if TRAIN_STAGE == "sft" else PRETRAIN_VAL_TOKENS_PATH
TRAIN_MASK_PATH = SFT_TRAIN_MASK_PATH if TRAIN_STAGE == "sft" else None
VAL_MASK_PATH = SFT_VAL_MASK_PATH if TRAIN_STAGE == "sft" else None
DATA_DOCUMENT_MODE = "blankline" if TRAIN_STAGE == "sft" else "line"
USE_LOSS_MASK = TRAIN_STAGE == "sft"

BASE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model", "SolenaV2.eqx")
SFT_CHECKPOINT_PATH  = os.path.join(CHECKPOINT_DIR, "model", "SolenaV2-sft.eqx")
CHECKPOINT_PATH      = SFT_CHECKPOINT_PATH if TRAIN_STAGE == "sft" else BASE_CHECKPOINT_PATH
LOAD_CHECKPOINT_PATH = CHECKPOINT_PATH
if TRAIN_STAGE == "sft" and not os.path.exists(LOAD_CHECKPOINT_PATH):
    LOAD_CHECKPOINT_PATH = BASE_CHECKPOINT_PATH

TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "tokenizer", "solena.model")

RESUME         = os.getenv("SOLENA_RESUME", "1") != "0"
SAVE_BEST_ONLY = True
MAX_ATTENTION_MATRIX_MB = float(os.getenv("SOLENA_MAX_ATTENTION_MATRIX_MB", "24"))


def attention_matrix_mb() -> float:
    dtype_bytes = 2 if PARAM_DTYPE == "bfloat16" else 4
    return N_HEADS * SEQ_LEN * SEQ_LEN * dtype_bytes / (1024 * 1024)


def validate_config() -> None:
    if EMBED_DIM % N_HEADS != 0:
        raise ValueError("EMBED_DIM must be divisible by N_HEADS")
    if NUM_DEVICES * PER_DEVICE_BATCH_SIZE != BATCH_SIZE:
        raise ValueError("BATCH_SIZE must equal NUM_DEVICES * PER_DEVICE_BATCH_SIZE")
    if LOGIT_CHUNK_SIZE <= 0:
        raise ValueError("LOGIT_CHUNK_SIZE must be positive")
    if TRAIN_PREFETCH_BATCHES < 0:
        raise ValueError("TRAIN_PREFETCH_BATCHES must be >= 0")
    if TRAIN_LOSS_SYNC_INTERVAL <= 0:
        raise ValueError("TRAIN_LOSS_SYNC_INTERVAL must be positive")
    if not 0 < VAL_RATIO < 1:
        raise ValueError("VAL_RATIO must be between 0 and 1")
    if OPTIMIZER not in {"adamw", "adafactor"}:
        raise ValueError(f"unknown OPTIMIZER: {OPTIMIZER}")
    if WEIGHT_DECAY < 0:
        raise ValueError("WEIGHT_DECAY must be >= 0")
    if PARAM_DTYPE not in {"float32", "bfloat16"}:
        raise ValueError(f"unknown PARAM_DTYPE: {PARAM_DTYPE}")
    if GEN_TEMPERATURE <= 0:
        raise ValueError("GEN_TEMPERATURE must be > 0")
    if abs(sum(PRETRAIN_MIX.values()) - 1.0) > 1e-6:
        raise ValueError("PRETRAIN_MIX weights must sum to 1.0")
    if USE_MESH and NUM_DEVICES <= 1:
        raise ValueError("USE_MESH requires NUM_DEVICES > 1")
    if (
        PROFILE in {"kaggle_tpu_8", "trc_tpu_16", "trc_tpu_64", "tpu_train"}
        and attention_matrix_mb() > MAX_ATTENTION_MATRIX_MB
    ):
        raise ValueError(
            f"attention score tensor is likely too large for TPU vmem: "
            f"{attention_matrix_mb():.1f}MB; lower SEQ_LEN/N_HEADS or set SOLENA_MAX_ATTENTION_MATRIX_MB"
        )


validate_config()
