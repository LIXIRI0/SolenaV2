import jax
import jax.numpy as jnp
from models.SolenaV2 import SolenaV2
from config import VOCAB_SIZE, SEQ_LEN, EMBED_DIM, N_HEADS, N_LAYERS, FF_DIM, DROPOUT, BATCH_SIZE

key = jax.random.PRNGKey(0)
model = SolenaV2(key, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS, ff_dim=FF_DIM, dropout=DROPOUT)
x=jnp.ones((BATCH_SIZE, SEQ_LEN), dtype=jnp.int32)
logits = model(x)
print(logits.shape) 