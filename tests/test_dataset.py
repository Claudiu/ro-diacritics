from pathlib import Path

import numpy as np

from diacritics.corpus.source import Document
from diacritics.dataset.shards import INPUT, LABEL, open_split, write_split
from diacritics.domain.alphabet import PAD_ID, Alphabet
from diacritics.domain.label import Label


def test_shards_round_trip_with_padding(tmp_path: Path) -> None:
    alphabet = Alphabet(tuple("ăcs "))
    docs = [Document("1", "casă casă"), Document("2", "c")]
    meta = write_split(tmp_path / "train.bin", iter(docs), alphabet, window=4, stride=3)

    rows = open_split(tmp_path, "train")

    assert meta.rows == 4
    assert rows.shape == (4, 4, 2)
    assert rows[0, :, INPUT].tolist() == alphabet.encode("casa")
    assert rows[0, :, LABEL].tolist() == [0, 0, 0, int(Label.BREVE_COMMA)]
    assert rows[2, :, INPUT].tolist() == alphabet.encode("asa") + [PAD_ID]
    assert rows[2, :, LABEL].tolist() == [0, 0, 1, int(Label.NONE)]
    assert rows[3, :, INPUT].tolist() == alphabet.encode("c") + [PAD_ID] * 3
    assert np.all(rows[3, 1:, LABEL] == int(Label.NONE))
