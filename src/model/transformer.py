"""
HookedTransformer factory for Raga sequence modeling.

Small transformer (~200K params) for autoregressive next-swara prediction.
Uses TransformerLens for mechanistic interpretability hooks.
"""

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from ..data.tokenizer import SwaraTokenizer


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def create_raga_transformer(
    tokenizer: SwaraTokenizer,
    n_layers: int = 4,
    n_heads: int = 4,
    d_model: int = 128,
    d_mlp: int = 512,
    max_seq_len: int = 72,
    act_fn: str = "gelu",
    device: str = None,
) -> HookedTransformer:
    """
    Create a small HookedTransformer for Raga sequence modeling.

    Default config: ~200K params, trainable in minutes on MacBook.
    """
    if device is None:
        device = _select_device()

    d_head = d_model // n_heads

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=d_mlp,
        d_vocab=tokenizer.vocab_size,
        n_ctx=max_seq_len,
        act_fn=act_fn,
        positional_embedding_type="standard",
        normalization_type="LN",
        attn_only=False,
        device=device,
        seed=42,
        init_weights=True,
    )

    model = HookedTransformer(cfg)
    return model
