import equinox as eqx
import jax
import jax.numpy as jnp

from config import DROPOUT, EMBED_DIM, FF_DIM, N_HEADS, N_LAYERS, PARAM_DTYPE, SEQ_LEN, USE_REMAT, VOCAB_SIZE


def _param_dtype():
    if PARAM_DTYPE == "bfloat16":
        return jnp.bfloat16
    if PARAM_DTYPE == "float32":
        return jnp.float32
    raise ValueError(f"unknown PARAM_DTYPE: {PARAM_DTYPE}")


def _init_weight(key: jax.Array, shape: tuple[int, ...], scale: float = 0.02) -> jax.Array:
    return (jax.random.normal(key, shape) * scale).astype(_param_dtype())


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
        self.weight = jnp.ones((dim,), dtype=_param_dtype())
        self.bias = jnp.zeros((dim,), dtype=_param_dtype())

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        out = self.weight.astype(jnp.float32) * (x - mean) * jax.lax.rsqrt(var + self.eps)
        out = out + self.bias.astype(jnp.float32)
        return out.astype(dtype)


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
        self.qkv_b = jnp.zeros((3 * embed_dim,), dtype=_param_dtype())
        self.out_w = _init_weight(out_key, (embed_dim, embed_dim))
        self.out_b = jnp.zeros((embed_dim,), dtype=_param_dtype())
        self.n_heads = n_heads
        self.dropout = dropout

    def __call__(self, x: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
        batch, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.n_heads

        qkv = x @ self.qkv_w + self.qkv_b
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        out = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=head_dim**-0.5,
            is_causal=True,
        )
        out = _dropout(out, key, self.dropout, train)
        out = out.reshape(batch, seq_len, embed_dim)
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
        self.b1 = jnp.zeros((ff_dim,), dtype=_param_dtype())
        self.w2 = _init_weight(key2, (ff_dim, embed_dim))
        self.b2 = jnp.zeros((embed_dim,), dtype=_param_dtype())
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

    def hidden_states(self, idx: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
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

        return self.ln_f(x)

    def __call__(self, idx: jax.Array, key: jax.Array | None = None, train: bool = False) -> jax.Array:
        return self.hidden_states(idx, key=key, train=train) @ self.token_embedding.T

    def logits_at(self, idx: jax.Array, position: jax.Array, key: jax.Array | None = None) -> jax.Array:
        hidden = self.hidden_states(idx, key=key, train=False)
        return hidden[:, position] @ self.token_embedding.T
