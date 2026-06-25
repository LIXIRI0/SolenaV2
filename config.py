import os

PROFILE = "collab_tpu"  # "cpu_dev", "cpu_full", "collab_tpu", "tpu_train"
TRAIN_STAGE = "sft"     # "pretrain" or "sft"

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

if TRAIN_STAGE == "sft":
    LR = 1e-5
    EPOCHS_PER_RUN = 3
elif TRAIN_STAGE != "pretrain":
    raise ValueError(f"unknown TRAIN_STAGE: {TRAIN_STAGE}")

GEN_MAX_NEW_TOKENS = 160
GEN_MIN_NEW_TOKENS = 24
GEN_TEMPERATURE    = 0.8
GEN_TOP_P          = 0.92
GEN_TOP_K          = 50
GEN_REPETITION_PENALTY = 1.15
GEN_REPETITION_WINDOW  = 128
GEN_NO_REPEAT_NGRAM_SIZE = 4
GEN_MAX_BANNED_TOKENS = 128
GEN_STOP_AFTER_SENTENCE = True
GEN_SEED           = 0
GEN_SHOW_FULL_TEXT = False
GEN_PROMPT_MODE    = "chat" if TRAIN_STAGE == "sft" else "plain"
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

SFT_PERSONA_NAME = "Solena"
SFT_CREATOR_NAME = "Lixiri"
SFT_PERSONA_REPEATS = 192
SFT_PERSONA_STYLE = (
    "helpful, warm, direct, lightly playful, and honest when unsure"
)

DROPOUT = 0.1

ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
PRETRAIN_DATA_PATH = os.path.join(ROOT_DIR, "data", "raw.txt")
SFT_DATA_PATH      = os.path.join(ROOT_DIR, "data", "sft_raw.txt")

PRETRAIN_TRAIN_TOKENS_PATH = os.path.join(ROOT_DIR, "data", "train.npy")
PRETRAIN_VAL_TOKENS_PATH   = os.path.join(ROOT_DIR, "data", "val.npy")
SFT_TRAIN_TOKENS_PATH      = os.path.join(ROOT_DIR, "data", "sft_train.npy")
SFT_VAL_TOKENS_PATH        = os.path.join(ROOT_DIR, "data", "sft_val.npy")
SFT_TRAIN_MASK_PATH        = os.path.join(ROOT_DIR, "data", "sft_train_mask.npy")
SFT_VAL_MASK_PATH          = os.path.join(ROOT_DIR, "data", "sft_val_mask.npy")

DATA_PATH = SFT_DATA_PATH if TRAIN_STAGE == "sft" else PRETRAIN_DATA_PATH
TRAIN_TOKENS_PATH = SFT_TRAIN_TOKENS_PATH if TRAIN_STAGE == "sft" else PRETRAIN_TRAIN_TOKENS_PATH
VAL_TOKENS_PATH = SFT_VAL_TOKENS_PATH if TRAIN_STAGE == "sft" else PRETRAIN_VAL_TOKENS_PATH
TRAIN_MASK_PATH = SFT_TRAIN_MASK_PATH if TRAIN_STAGE == "sft" else None
VAL_MASK_PATH = SFT_VAL_MASK_PATH if TRAIN_STAGE == "sft" else None
DATA_DOCUMENT_MODE = "blankline" if TRAIN_STAGE == "sft" else "line"
USE_LOSS_MASK = TRAIN_STAGE == "sft"

BASE_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "checkpoints", "model", "SolenaV2.eqx")
SFT_CHECKPOINT_PATH  = os.path.join(ROOT_DIR, "checkpoints", "model", "SolenaV2-sft.eqx")
CHECKPOINT_PATH      = SFT_CHECKPOINT_PATH if TRAIN_STAGE == "sft" else BASE_CHECKPOINT_PATH
LOAD_CHECKPOINT_PATH = CHECKPOINT_PATH
if TRAIN_STAGE == "sft" and not os.path.exists(LOAD_CHECKPOINT_PATH):
    LOAD_CHECKPOINT_PATH = BASE_CHECKPOINT_PATH

TOKENIZER_PATH = os.path.join(ROOT_DIR, "checkpoints", "tokenizer", "solena.model")

RESUME         = True
SAVE_BEST_ONLY = True
