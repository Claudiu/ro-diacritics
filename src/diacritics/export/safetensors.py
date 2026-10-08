"""The contract between training and inference: model.safetensors + config.json.

crates/model reads exactly these two files; change both sides together.
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
FORMAT_VERSION = 1


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
    # fp16 halves the browser download; both loaders upcast to f32, accuracy is unchanged.
    tensors = {name: t.detach().cpu().half().contiguous() for name, t in model.state_dict().items()}
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
    model.load_state_dict(load(raw_weights))
    model.eval()
    alphabet = Alphabet.from_dict(data["alphabet"])

    return model, alphabet, int(data["overlap"]), float(data["threshold"])


def checkpoint_bytes(state: dict[str, Any]) -> bytes:
    import io

    buffer = io.BytesIO()
    torch.save(state, buffer)

    return buffer.getvalue()


def load_checkpoint(raw: bytes) -> dict[str, Any]:
    import io

    state: dict[str, Any] = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)

    return state
