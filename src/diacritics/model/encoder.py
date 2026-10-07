"""Character-level transformer encoder.

Written out layer by layer (no nn.TransformerEncoder) so parameter names map 1:1 onto
crates/model/src/transformer.rs. Pre-LayerNorm blocks, learned positions, GELU.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Self

import torch
import torch.nn.functional as f
from torch import Tensor, nn

from diacritics.domain.alphabet import PAD_ID
from diacritics.domain.label import Label


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int
    window: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    n_classes: int = len(Label)

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            vocab_size=int(data["vocab_size"]),
            window=int(data["window"]),
            d_model=int(data["d_model"]),
            n_layers=int(data["n_layers"]),
            n_heads=int(data["n_heads"]),
            d_ff=int(data["d_ff"]),
            dropout=float(data["dropout"]),
        )


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        """`allowed` is (batch, 1, 1, t) bool: which keys may be attended to."""
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(b, t, self.n_heads, -1).transpose(1, 2)
        k = k.view(b, t, self.n_heads, -1).transpose(1, 2)
        v = v.view(b, t, self.n_heads, -1).transpose(1, 2)
        y = f.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=self.dropout if self.training else 0
        )
        y = y.transpose(1, 2).reshape(b, t, d)
        out: Tensor = self.out(y)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.down(f.gelu(self.up(x)))

        return out


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        x = x + self.drop(self.attn(self.ln1(x), allowed))
        x = x + self.drop(self.ff(self.ln2(x)))

        return x


class CharEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.window, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)
        self.drop = nn.Dropout(cfg.dropout)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids: Tensor) -> Tensor:
        """(batch, t) int64 ids → (batch, t, n_classes) logits; t <= window.

        PAD_ID positions are masked out as keys, so padding never changes the output of
        real positions and inputs only need padding to their own length."""
        t = ids.shape[1]
        if t > self.cfg.window:
            raise ValueError(f"sequence of {t} exceeds window {self.cfg.window}")

        allowed = (ids != PAD_ID)[:, None, None, :]
        positions = torch.arange(t, device=ids.device)
        x = self.drop(self.embed(ids) + self.pos(positions))
        for block in self.blocks:
            x = block(x, allowed)

        logits: Tensor = self.head(self.ln_f(x))

        return logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def candidate_loss(logits: Tensor, labels: Tensor, candidates: Tensor) -> Tensor:
    """Cross-entropy over candidate positions only; the rest can never carry a diacritic."""
    picked = candidates.view(-1)
    flat_logits = logits.view(-1, logits.shape[-1])[picked]
    flat_labels = labels.view(-1)[picked]
    if flat_labels.numel() == 0:
        return logits.sum() * 0

    return f.cross_entropy(flat_logits, flat_labels)


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 10% of base."""
    if step < warmup:
        return base_lr * (step + 1) / warmup

    progress = min(1.0, (step - warmup) / max(1, total - warmup))

    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
