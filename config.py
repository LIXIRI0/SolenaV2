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
    SEQ_LEN        = 256
    BATCH_SIZE     = 8
    EMBED_DIM      = 384
    N_HEADS        = 6
    N_LAYERS       = 6
    FF_DIM         = 1536
    LR             = 2e-4
    MAX_BATCHES    = None
    VAL_BATCHES    = 20
    EPOCHS_PER_RUN = 20
    MAX_EPOCHS     = None

elif PROFILE == "tpu_train":
    VOCAB_SIZE     = 32000
    SEQ_LEN        = 1024
    BATCH_SIZE     = 32      # per chip
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

GEN_MAX_NEW_TOKENS = 200
GEN_TEMPERATURE    = 0.7
GEN_TOP_P          = 0.95
GEN_TOP_K          = None
GEN_SEED           = 0
GEN_SHOW_FULL_TEXT = False
GEN_EXIT_COMMANDS  = ("exit", "quit", "q")

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
