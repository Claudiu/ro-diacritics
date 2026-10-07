"""Training loop: resumable from the latest checkpoint, exports the best model."""

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from diacritics.artifacts.store import ArtifactStore
from diacritics.common.logging import log
from diacritics.common.progress import Progress
from diacritics.config.settings import Settings
from diacritics.dataset.shards import INPUT, LABEL, open_split
from diacritics.domain.alphabet import Alphabet
from diacritics.domain.normalize import CANDIDATES
from diacritics.export.safetensors import checkpoint_bytes, export, load_checkpoint
from diacritics.metrics.candidates import CandidateMetrics
from diacritics.model.encoder import CharEncoder, ModelConfig, candidate_loss, lr_at

logger = logging.getLogger(__name__)

CHECKPOINT = "checkpoints/latest.pt"
BEST_EXPORT = "export"
STATS_EVERY = 50
"""Steps between loss reads: `.item()` waits for the GPU, so not every step."""


class WindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, rows: npt.NDArray[np.uint8]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        row = np.asarray(self._rows[idx])

        return torch.from_numpy(row[:, INPUT].astype(np.int64)), torch.from_numpy(
            row[:, LABEL].astype(np.int64)
        )


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def candidate_mask(alphabet: Alphabet, device: torch.device) -> Tensor:
    """(vocab,) bool: which input ids can carry a diacritic."""
    mask = torch.zeros(alphabet.size, dtype=torch.bool)
    for ch in CANDIDATES:
        (idx,) = alphabet.encode(ch)
        mask[idx] = True

    return mask.to(device)


@dataclass(slots=True)
class Trainer:
    settings: Settings
    alphabet: Alphabet
    store: ArtifactStore
    progress: Progress

    def run(self) -> None:
        s = self.settings
        device = pick_device(s.training.device)
        torch.manual_seed(s.training.seed)
        cfg = ModelConfig(
            vocab_size=self.alphabet.size,
            window=s.model.window,
            d_model=s.model.d_model,
            n_layers=s.model.n_layers,
            n_heads=s.model.n_heads,
            d_ff=s.model.d_ff,
            dropout=s.model.dropout,
        )
        model = CharEncoder(cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=s.training.lr, weight_decay=0.01)
        is_candidate = candidate_mask(self.alphabet, device)
        step = 0
        best = -1.0

        raw = self.store.read(CHECKPOINT)
        if raw is not None:
            state = load_checkpoint(raw)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            step = int(state["step"])
            best = float(state["best"])
            log(logger, "resumed", step=step, best=best)

        log(
            logger,
            "training",
            device=str(device),
            parameters=model.parameter_count(),
            config=cfg.to_dict(),
        )
        train_loader = self._loader("train", shuffle=True)
        valid_loader = self._loader("valid", shuffle=False)
        running_loss = torch.zeros((), device=device)
        running_steps = 0

        with self.progress.track("train", s.training.max_steps, completed=step) as bar:
            if best >= 0:
                bar.update(step, best=best)
            while step < s.training.max_steps:
                for ids, labels in train_loader:
                    if step >= s.training.max_steps:
                        break

                    lr = lr_at(step, s.training.lr, s.training.warmup_steps, s.training.max_steps)
                    loss = self._step(model, optimizer, ids, labels, lr, is_candidate, device)
                    running_loss += loss.detach()
                    running_steps += 1
                    step += 1

                    if step % STATS_EVERY == 0:
                        mean_loss = (running_loss / running_steps).item()
                        running_loss.zero_()
                        running_steps = 0
                        bar.update(step, loss=mean_loss, lr=lr)
                    else:
                        bar.update(step)

                    if step % s.training.eval_every == 0:
                        best = self._eval_and_export(
                            model, valid_loader, is_candidate, device, step, best
                        )
                        bar.update(step, best=best)
                    if step % s.training.checkpoint_every == 0:
                        self._checkpoint(model, optimizer, step, best)

        if step % s.training.checkpoint_every:
            self._checkpoint(model, optimizer, step, best)
        log(logger, "done", step=step, best=best)

    def _step(
        self,
        model: CharEncoder,
        optimizer: torch.optim.Optimizer,
        ids: Tensor,
        labels: Tensor,
        lr: float,
        is_candidate: Tensor,
        device: torch.device,
    ) -> Tensor:
        ids, labels = ids.to(device), labels.to(device)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids)
            loss = candidate_loss(logits.float(), labels, is_candidate[ids])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss

    def _eval_and_export(
        self,
        model: CharEncoder,
        loader: DataLoader[tuple[Tensor, Tensor]],
        is_candidate: Tensor,
        device: torch.device,
        step: int,
        best: float,
    ) -> float:
        """Evaluate, export when it beats `best`, and return the new best."""
        s = self.settings
        metrics = self.evaluate(model, loader, is_candidate, device)
        log(logger, "eval", step=step, **metrics)
        if metrics["candidate_accuracy"] <= best:
            return best

        export(self.store, BEST_EXPORT, model, self.alphabet, s.model.overlap, s.training.threshold)
        log(logger, "exported best", step=step, candidate_accuracy=metrics["candidate_accuracy"])

        return metrics["candidate_accuracy"]

    def evaluate(
        self,
        model: CharEncoder,
        loader: DataLoader[tuple[Tensor, Tensor]],
        is_candidate: Tensor,
        device: torch.device,
    ) -> dict[str, float]:
        model.eval()
        metrics = CandidateMetrics()
        batches = min(self.settings.training.eval_batches, len(loader))
        with torch.no_grad(), self.progress.track("eval", batches, transient=True) as bar:
            for i, (ids, labels) in enumerate(loader):
                if i >= batches:
                    break

                ids, labels = ids.to(device), labels.to(device)
                pred = model(ids).argmax(dim=-1)
                mask = is_candidate[ids]
                metrics.add(labels[mask].tolist(), pred[mask].tolist())
                bar.update(i + 1)

        return metrics.summary()

    def _loader(self, split: str, shuffle: bool) -> DataLoader[tuple[Tensor, Tensor]]:
        t = self.settings.training

        return DataLoader(
            WindowDataset(open_split(self.settings.paths.processed, split)),
            batch_size=t.batch_size,
            shuffle=shuffle,
            num_workers=t.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle,
            persistent_workers=t.num_workers > 0,
        )

    def _checkpoint(
        self, model: CharEncoder, optimizer: torch.optim.Optimizer, step: int, best: float
    ) -> None:
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best": best,
        }
        self.store.write(CHECKPOINT, checkpoint_bytes(state))
        log(logger, "checkpoint", step=step)
