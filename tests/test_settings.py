import pytest

from diacritics.config.settings import Settings


def test_defaults_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIACRITICS_D_MODEL", raising=False)
    settings = Settings.from_env()

    assert settings.model.d_model == 192
    assert settings.paths.processed.name == "processed"


def test_env_overrides_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIACRITICS_D_MODEL", "100")
    monkeypatch.setenv("DIACRITICS_N_HEADS", "3")
    with pytest.raises(ValueError, match="N_HEADS"):
        Settings.from_env()

    monkeypatch.setenv("DIACRITICS_N_HEADS", "4")
    assert Settings.from_env().model.d_model == 100

    monkeypatch.setenv("DIACRITICS_N_HEADS", "four")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()
