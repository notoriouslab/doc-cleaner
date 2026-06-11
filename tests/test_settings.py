"""Unit tests for the GUI preferences layer (macapp/settings.py, D1)."""

import json
import os

import pytest

from macapp import settings


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point the settings module at an isolated app-data dir under tmp_path."""
    d = tmp_path / "appdata"
    monkeypatch.setattr(settings, "app_data_dir", lambda: str(d))
    return d


def test_defaults_shape():
    assert settings.DEFAULTS == {
        "version": settings.SCHEMA_VERSION,
        "output_mode": "sibling",
        "custom_output_dir": None,
        "last_input_dir": None,
    }


def test_load_missing_returns_defaults(data_dir):
    assert not data_dir.exists()
    assert settings.load() == settings.DEFAULTS


def test_round_trip(data_dir):
    settings.save(
        {
            "version": 1,
            "output_mode": "custom",
            "custom_output_dir": "/tmp/out",
            "last_input_dir": "/tmp/in",
        }
    )
    loaded = settings.load()
    assert loaded["output_mode"] == "custom"
    assert loaded["custom_output_dir"] == "/tmp/out"
    assert loaded["last_input_dir"] == "/tmp/in"


def test_save_writes_under_app_data_dir(data_dir):
    settings.save(settings.DEFAULTS)
    written = data_dir / "settings.json"
    assert written.is_file()
    # No temp leftovers from the atomic write.
    leftovers = [p for p in os.listdir(data_dir) if p.startswith(".settings-")]
    assert leftovers == []


def test_save_only_persists_known_keys(data_dir):
    settings.save({"output_mode": "desktop", "bogus": 123})
    raw = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
    assert "bogus" not in raw
    assert set(raw) == set(settings.DEFAULTS)
    assert raw["output_mode"] == "desktop"


def test_corrupt_json_returns_defaults(data_dir):
    data_dir.mkdir(parents=True)
    (data_dir / "settings.json").write_text("{not valid json", encoding="utf-8")
    assert settings.load() == settings.DEFAULTS  # no raise


def test_non_dict_root_returns_defaults(data_dir):
    data_dir.mkdir(parents=True)
    (data_dir / "settings.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert settings.load() == settings.DEFAULTS  # no raise


def test_bad_typed_value_falls_back_per_key(data_dir):
    data_dir.mkdir(parents=True)
    (data_dir / "settings.json").write_text(
        json.dumps(
            {
                "output_mode": "not_a_mode",   # invalid -> default "sibling"
                "custom_output_dir": 12345,      # invalid -> default None
                "last_input_dir": "/keep/me",    # valid -> kept
            }
        ),
        encoding="utf-8",
    )
    loaded = settings.load()
    assert loaded["output_mode"] == "sibling"
    assert loaded["custom_output_dir"] is None
    assert loaded["last_input_dir"] == "/keep/me"


def test_empty_object_returns_defaults(data_dir):
    data_dir.mkdir(parents=True)
    (data_dir / "settings.json").write_text("{}", encoding="utf-8")
    assert settings.load() == settings.DEFAULTS


def test_app_data_dir_is_platform_path():
    # Smoke check: returns a non-empty path ending in the app name.
    p = settings.app_data_dir()
    assert p.endswith("Doc Cleaner")
