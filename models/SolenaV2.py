import math
import equinox as eqx
import jax
import jax.numpy as jnp

from config import DROPOUT, EMBED_DIM, FF_DIM, N_HEADS, N_LAYERS, SEQ_LEN, USE_REMAT, VOCAB_SIZE


def _init_weight(key: jax.Array, shape: tuple[int, ...], scale: float = 0.02) -> jax.Array:
    return jax.random.normal(key, shape) * scale


def _dropout(x: jax.Array, key: jax.Array | None, rate: float, train: bool) -> jax.Array:
    if not train or key is None or rate == 0.0:
        return x
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, x.shape)
    return jnp.where(mask, x / keep, 0)


class LayerNorm(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    eps: float = 1e-5

    def __init__(self, dim: int):
        self.weight = jnp.ones((dim,))
        self.bias = jnp.zeros((dim,))

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return self.weight * (x - mean) * jax.lax.rsqrt(var + self.eps) + self.bias


class CausalSelfAttention(eqx.Module):
    qkv_w: jax.Array
    qkv_b: jax.Array
    out_w: jax.Array
    out_b: jax.Array
    n_heads: int
    dropout: float

    def __init__(self, key: jax.Array, embed_dim: int, n_heads: int, dropout: float):
        if embed_dim % n_heads != 0:
            raise ValueError("EMBED_DIM must be divisible by N_HEADS")

        qkv_key, out_key = jax.random.split(key)
        self.qkv_w = _init_weight(qkv_key, (embed_dim, 3 * embed_dim))
        self.qkv_b = jnp.zeros((3 * embed_dim,))
        self.out_w = _init_weight(out_key, (embed_dim, embed_dim))
        self.out_b = jnp.zeros((embed_dim,))
        self.n_heads = n_heads
        self.dropout = dropout

    def __call__(self, x: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
        batch, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.n_heads

        qkv = x @ self.qkv_w + self.qkv_b
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ jnp.swapaxes(k, -1, -2)) / math.sqrt(head_dim)
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(mask[None, None, :, :], scores, -jnp.inf)

        weights = jax.nn.softmax(scores, axis=-1)
        weights = _dropout(weights, key, self.dropout, train)

        out = weights @ v
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch, seq_len, embed_dim)
        return out @ self.out_w + self.out_b


class MLP(eqx.Module):
    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    dropout: float

    def __init__(self, key: jax.Array, embed_dim: int, ff_dim: int, dropout: float):
        key1, key2 = jax.random.split(key)
        self.w1 = _init_weight(key1, (embed_dim, ff_dim))
        self.b1 = jnp.zeros((ff_dim,))
        self.w2 = _init_weight(key2, (ff_dim, embed_dim))
        self.b2 = jnp.zeros((embed_dim,))
        self.dropout = dropout

    def __call__(self, x: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
        x = jax.nn.gelu(x @ self.w1 + self.b1)
        x = _dropout(x, key, self.dropout, train)
        return x @ self.w2 + self.b2


class TransformerBlock(eqx.Module):
    ln1: LayerNorm
    attn: CausalSelfAttention
    ln2: LayerNorm
    mlp: MLP

    def __init__(self, key: jax.Array, embed_dim: int, n_heads: int, ff_dim: int, dropout: float):
        attn_key, mlp_key = jax.random.split(key)
        self.ln1 = LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(attn_key, embed_dim, n_heads, dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.mlp = MLP(mlp_key, embed_dim, ff_dim, dropout)

    def __call__(self, x: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
        attn_key, mlp_key = (None, None) if key is None else jax.random.split(key)
        x = x + self.attn(self.ln1(x), key=attn_key, train=train)
        x = x + self.mlp(self.ln2(x), key=mlp_key, train=train)
        return x


@eqx.filter_checkpoint
def _checkpointed_block(block: TransformerBlock, x: jax.Array, key: jax.Array | None) -> jax.Array:
    return block(x, key=key, train=True)


class SolenaV2(eqx.Module):
    token_embedding: jax.Array
    pos_embedding: jax.Array
    blocks: tuple[TransformerBlock, ...]
    ln_f: LayerNorm

    def __init__(
        self,
        key: jax.Array,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        embed_dim: int = EMBED_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ):
        embed_key, pos_key, *block_keys = jax.random.split(key, n_layers + 2)
        self.token_embedding = _init_weight(embed_key, (vocab_size, embed_dim))
        self.pos_embedding = _init_weight(pos_key, (seq_len, embed_dim))
        self.blocks = tuple(
            TransformerBlock(block_keys[i], embed_dim, n_heads, ff_dim, dropout)
            for i in range(n_layers)
        )
        self.ln_f = LayerNorm(embed_dim)

    def __call__(self, idx: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
        _, seq_len = idx.shape
        if seq_len > self.pos_embedding.shape[0]:
            raise ValueError("input sequence is longer than SEQ_LEN")

        x = self.token_embedding[idx] + self.pos_embedding[:seq_len]
        block_keys = [None] * len(self.blocks) if key is None else list(jax.random.split(key, len(self.blocks)))

        for block, block_key in zip(self.blocks, block_keys):
            if train and USE_REMAT:
                x = _checkpointed_block(block, x, block_key)
            else:
                x = block(x, key=block_key, train=train)

        x = self.ln_f(x)
        return x @ self.token_embedding.T
