import os

PROFILE = "collab_tpu"  # "cpu_dev", "cpu_full", "collab_tpu", "tpu_train"

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
    LR             = 1.5e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 30
    EPOCHS_PER_RUN = 50
    MAX_EPOCHS     = None

elif PROFILE == "tpu_train":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 1024
    NUM_DEVICES    = 8
    PER_DEVICE_BATCH_SIZE = 4
    BATCH_SIZE     = NUM_DEVICES * PER_DEVICE_BATCH_SIZE
    EMBED_DIM      = 1024
    N_HEADS        = 16
    N_LAYERS       = 24
    FF_DIM         = 4096
    LR             = 1.5e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 50
    EPOCHS_PER_RUN = 100
    MAX_EPOCHS     = None
    GRAD_ACCUM_STEPS = 4

else:
    raise ValueError(f"unknown PROFILE: {PROFILE}")

if "NUM_DEVICES" not in globals():
    NUM_DEVICES = 1
if "PER_DEVICE_BATCH_SIZE" not in globals():
    PER_DEVICE_BATCH_SIZE = BATCH_SIZE
USE_DATA_PARALLEL = NUM_DEVICES > 1

GEN_MAX_NEW_TOKENS = 300
GEN_TEMPERATURE    = 0.8
GEN_TOP_P          = 0.92
GEN_TOP_K          = None
GEN_SEED           = 0
GEN_SHOW_FULL_TEXT = False
GEN_EXIT_COMMANDS  = ("exit", "quit", "q")

PRETRAIN_TARGET_TOKENS = 300_000_000
PRETRAIN_CHARS_PER_TOKEN = 4
PRETRAIN_SEED = 42
PRETRAIN_HF_TIMEOUT = 60
PRETRAIN_SHUFFLE_BUFFER_SIZE = 10_000
PRETRAIN_MIX = {
    "clean_web": 0.70,
    "wiki": 0.15,
    "stories": 0.10,
    "cs": 0.05,
}

DROPOUT = 0.1

ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(ROOT_DIR, "data", "raw.txt")
VAL_PATH         = os.path.join(ROOT_DIR, "data", "val.txt")
TRAIN_TOKENS_PATH = os.path.join(ROOT_DIR, "data", "train.npy")
VAL_TOKENS_PATH   = os.path.join(ROOT_DIR, "data", "val.npy")
CHECKPOINT_PATH  = os.path.join(ROOT_DIR, "checkpoints", "model", "SolenaV2.eqx")
TOKENIZER_PATH   = os.path.join(ROOT_DIR, "checkpoints", "tokenizer", "solena.model")

RESUME         = True
SAVE_BEST_ONLY = True
