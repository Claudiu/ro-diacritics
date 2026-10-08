import torch

from diacritics.artifacts.memory import InMemoryStore
from diacritics.domain.alphabet import Alphabet
from diacritics.export.safetensors import export, load_export
from diacritics.model.encoder import CharEncoder, ModelConfig, candidate_loss, lr_at
from diacritics.model.infer import Predictor

ALPHABET = Alphabet(tuple(" .,abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
CFG = ModelConfig(
    vocab_size=ALPHABET.size, window=16, d_model=16, n_layers=2, n_heads=2, d_ff=32, dropout=0.0
)


def test_forward_shape() -> None:
    model = CharEncoder(CFG)
    logits = model(torch.zeros(3, 10, dtype=torch.long))

    assert logits.shape == (3, 10, 3)
    assert model.parameter_count() > 0


def test_candidate_loss_only_counts_candidates() -> None:
    logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    labels = torch.tensor([[0, 0]])

    confident_wrong = candidate_loss(logits, labels, torch.tensor([[False, True]]))
    confident_right = candidate_loss(logits, labels, torch.tensor([[True, False]]))
    none = candidate_loss(logits, labels, torch.tensor([[False, False]]))

    assert confident_wrong > 5
    assert confident_right < 0.01
    assert none.item() == 0.0


def test_lr_schedule_warms_up_then_decays() -> None:
    assert lr_at(0, 1.0, 10, 100) == 0.1
    assert lr_at(9, 1.0, 10, 100) == 1.0
    assert 0.1 < lr_at(50, 1.0, 10, 100) < 1.0
    assert abs(lr_at(100, 1.0, 10, 100) - 0.1) < 1e-9


def test_predictor_covers_long_text_and_round_trips_export() -> None:
    torch.manual_seed(0)
    model = CharEncoder(CFG).eval()
    predictor = Predictor(model, ALPHABET, overlap=4, threshold=0.0)
    text = "Langa casa mea e casa ta si e o casa foarte frumoasa, asa ca stam acasa."

    predictions = predictor.predict(text)
    restored = predictor.restore(text)

    assert len(predictions) == len(text)
    assert len(restored) == len(text)
    assert predictor.predict("") == []
    assert predictor.restore("") == ""

    store = InMemoryStore()
    export(store, "m", model, ALPHABET, overlap=4, threshold=0.25)
    loaded, alphabet, overlap, threshold = load_export(store, "m")

    assert (alphabet, overlap, threshold) == (ALPHABET, 4, 0.25)
    # Weights are stored as fp16, so confidences round-trip only to fp16 precision.
    reloaded = Predictor(loaded, alphabet, overlap, 0.0).predict(text)
    assert [p.label for p in reloaded] == [p.label for p in predictions]
    assert all(
        abs(a.confidence - b.confidence) < 1e-3 for a, b in zip(reloaded, predictions, strict=True)
    )
