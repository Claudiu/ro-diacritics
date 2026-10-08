"""The contract between training and inference: model.safetensors + config.json.

crates/model reads exactly these two files; change both sides together.

Matrices (linear weights, embeddings) are 4-bit: each group of GROUP consecutive values
shares one fp16 scale, stored as `<name>.scale`; values are q - 8 for q in 1..15, two per
byte, the even element in the low nibble, so `<name>` is uint8 with half the columns.
Vectors (biases, layer norms) are fp16. Both loaders rebuild f32 weights.
"""

import json
from typing import Any

import torch
from safetensors.torch import load, save

from diacritics.artifacts.store import ArtifactStore
from diacritics.domain.alphabet import Alphabet
from diacritics.domain.label import LABEL_NAMES
from diacritics.model.encoder import CharEncoder, ModelConfig

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
FORMAT_VERSION = 2
GROUP = 32
SCALE = ".scale"


def export_config(cfg: ModelConfig, alphabet: Alphabet, overlap: int, threshold: float) -> str:
    data: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "model": cfg.to_dict(),
        "alphabet": alphabet.to_dict(),
        "labels": list(LABEL_NAMES),
        "overlap": overlap,
        "threshold": threshold,
    }

    return json.dumps(data, ensure_ascii=False, indent=2)


def export(
    store: ArtifactStore,
    prefix: str,
    model: CharEncoder,
    alphabet: Alphabet,
    overlap: int,
    threshold: float,
) -> None:
    tensors: dict[str, torch.Tensor] = {}
    for name, t in model.state_dict().items():
        t = t.detach().cpu().float()
        if t.dim() == 2:
            tensors[name], tensors[name + SCALE] = quantize(t)
        else:
            tensors[name] = t.half().contiguous()
    store.write(f"{prefix}/{WEIGHTS_FILE}", save(tensors))
    store.write(
        f"{prefix}/{CONFIG_FILE}",
        export_config(model.cfg, alphabet, overlap, threshold).encode(),
    )


def load_export(store: ArtifactStore, prefix: str) -> tuple[CharEncoder, Alphabet, int, float]:
    """Model (on CPU, eval mode), alphabet, overlap and threshold from an export."""
    raw_config = store.read(f"{prefix}/{CONFIG_FILE}")
    raw_weights = store.read(f"{prefix}/{WEIGHTS_FILE}")
    if raw_config is None or raw_weights is None:
        raise FileNotFoundError(f"no export under {prefix!r}")

    data = json.loads(raw_config)
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported export format {data.get('format_version')}")

    cfg = ModelConfig.from_dict(data["model"])
    model = CharEncoder(cfg)
    model.load_state_dict(dequantize_all(load(raw_weights)))
    model.eval()
    alphabet = Alphabet.from_dict(data["alphabet"])

    return model, alphabet, int(data["overlap"]), float(data["threshold"])


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, cols) f32 → (rows, cols/2) packed uint8 and one fp16 scale per GROUP values."""
    rows, cols = w.shape
    if cols % GROUP:
        raise ValueError(f"{cols} columns is not a multiple of {GROUP}")

    groups = w.reshape(-1, GROUP)
    scale = (groups.abs().amax(dim=1) / 7).clamp(min=1e-8).half()
    q = (groups / scale.float()[:, None]).round().clamp(-7, 7).to(torch.uint8) + 8
    q = q.reshape(rows, cols)
    packed = q[:, 0::2] | (q[:, 1::2] << 4)

    return packed.contiguous(), scale


def dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, half = packed.shape
    q = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(rows, 2 * half)
    values = (q.float() - 8).reshape(-1, GROUP) * scale.float()[:, None]

    return values.reshape(rows, 2 * half)


def dequantize_all(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: dequantize(t, tensors[name + SCALE]) if t.dtype == torch.uint8 else t.float()
        for name, t in tensors.items()
        if not name.endswith(SCALE)
    }


def checkpoint_bytes(state: dict[str, Any]) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)

    return buffer.getvalue()


def load_checkpoint(raw: bytes) -> dict[str, Any]:
    import io

    state: dict[str, Any] = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    return state
