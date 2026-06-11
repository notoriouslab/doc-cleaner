"""End-to-end: the GUI bridge really converts files to the chosen folder.

Unlike test_macapp (which mocks _run_one), this drives the *real* conversion
engine (core._run_one with ai="none") through Api._run_batch and asserts the
Markdown file actually lands in the right place. Closest automated proxy for
the manual GUI walkthrough (task 6.2): custom-folder output, sibling fallback,
recursive folder drop, and the AI-off invariant — all with real files on disk.
"""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("webview", reason="pywebview not installed")


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path_factory, monkeypatch):
    from macapp import settings as _settings
    d = tmp_path_factory.mktemp("appdata")
    monkeypatch.setattr(_settings, "app_data_dir", lambda: str(d))


def _api():
    from macapp.app import Api
    api = Api()
    api._window = MagicMock()
    return api


def _make_txt(path, text="Hello world\n\nSecond paragraph of real content."):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_real_convert_to_custom_folder(tmp_path):
    """Two real .txt files from different folders convert into one chosen folder."""
    src1 = _make_txt(tmp_path / "folderA" / "doc1.txt")
    src2 = _make_txt(tmp_path / "folderB" / "doc2.txt")
    out = tmp_path / "chosen_output"
    out.mkdir()

    _api()._run_batch([src1, src2], "custom", str(out))

    produced = sorted(p.name for p in out.glob("*.md"))
    assert produced == ["doc1.md", "doc2.md"], f"got {produced}"
    assert "Hello world" in (out / "doc1.md").read_text(encoding="utf-8")


def test_real_convert_sibling_default(tmp_path):
    """Default sibling mode writes the .md next to the source."""
    src = _make_txt(tmp_path / "note.txt")
    _api()._run_batch([src], "sibling", None)
    assert (tmp_path / "note.md").is_file()


def test_stale_custom_folder_falls_back_to_sibling(tmp_path):
    """A deleted custom folder degrades to sibling output; file still produced + notice."""
    src = _make_txt(tmp_path / "src" / "a.txt")
    gone = str(tmp_path / "deleted_folder")  # never created
    api = _api()
    api._run_batch([src], "custom", gone)
    assert (tmp_path / "src" / "a.md").is_file()
    assert any("fallbackSibling" in str(c)
               for c in api._window.evaluate_js.call_args_list)


def test_recursive_drop_then_convert(tmp_path):
    """Dropping a folder collects nested files; converting lands every .md."""
    _make_txt(tmp_path / "tree" / "top.txt")
    _make_txt(tmp_path / "tree" / "sub" / "deep.txt")
    out = tmp_path / "out"
    out.mkdir()

    api = _api()
    from webview.dom import _dnd_state
    _dnd_state["paths"] = [(None, str(tmp_path / "tree"))]

    collected = api.get_dropped_paths()
    assert len(collected) == 2  # top.txt + sub/deep.txt found recursively

    api._run_batch(collected, "custom", str(out))
    produced = sorted(p.name for p in out.glob("*.md"))
    assert produced == ["deep.md", "top.md"], f"got {produced}"


def test_ai_stays_off(tmp_path, monkeypatch):
    """Phase A invariant: the GUI batch builds the env with ai='none'."""
    import macapp.app as appmod
    captured = {}
    real_build = appmod._core._build_env

    def _spy(*args, **kwargs):
        captured["ai"] = kwargs.get("ai", args[0] if args else None)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(appmod._core, "_build_env", _spy)
    src = _make_txt(tmp_path / "x.txt")
    _api()._run_batch([src], "sibling", None)
    assert captured["ai"] == "none"
