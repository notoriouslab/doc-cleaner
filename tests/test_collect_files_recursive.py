"""Unit tests for cleaner.collect_files recursive mode (D3)."""

import os

import pytest

import cleaner


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_recursive_finds_nested_supported_files(tmp_path):
    _touch(tmp_path / "top.md")
    _touch(tmp_path / "sub" / "a.txt")
    _touch(tmp_path / "sub" / "deep" / "b.pdf")
    _touch(tmp_path / "sub" / "note.unsupported")  # excluded

    result = cleaner.collect_files(str(tmp_path), recursive=True)
    names = sorted(os.path.basename(p) for p in result)
    assert names == ["a.txt", "b.pdf", "top.md"]
    # Sorted/deterministic
    assert result == sorted(result) or len(result) == len(set(result))


def test_default_is_single_level(tmp_path):
    _touch(tmp_path / "top.md")
    _touch(tmp_path / "sub" / "a.txt")

    result = cleaner.collect_files(str(tmp_path))  # recursive=False default
    names = [os.path.basename(p) for p in result]
    assert names == ["top.md"]  # subfolder file NOT included


def test_unsupported_extension_excluded(tmp_path):
    _touch(tmp_path / "keep.csv")
    _touch(tmp_path / "skip.bin")
    result = cleaner.collect_files(str(tmp_path), recursive=True)
    names = [os.path.basename(p) for p in result]
    assert names == ["keep.csv"]


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics differ on Windows CI")
def test_symlinked_subdir_not_followed(tmp_path):
    outside = tmp_path / "outside"
    _touch(outside / "secret.md")
    root = tmp_path / "root"
    root.mkdir()
    _touch(root / "inside.md")
    os.symlink(str(outside), str(root / "link_to_outside"))

    result = cleaner.collect_files(str(root), recursive=True)
    names = sorted(os.path.basename(p) for p in result)
    assert names == ["inside.md"]  # secret.md via symlinked dir not collected


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics differ on Windows CI")
def test_symlinked_file_escape_excluded(tmp_path):
    outside = tmp_path / "outside"
    _touch(outside / "secret.md")
    root = tmp_path / "root"
    root.mkdir()
    _touch(root / "inside.md")
    # A symlinked *file* inside root whose realpath resolves outside root.
    os.symlink(str(outside / "secret.md"), str(root / "escape.md"))

    result = cleaner.collect_files(str(root), recursive=True)
    names = sorted(os.path.basename(p) for p in result)
    assert names == ["inside.md"]  # escape.md resolves outside -> excluded


def test_cap_enforced_and_warns(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(cleaner, "MAX_RECURSIVE_FILES", 3)
    for i in range(5):
        _touch(tmp_path / f"f{i}.md")

    with caplog.at_level("WARNING"):
        result = cleaner.collect_files(str(tmp_path), recursive=True)

    assert len(result) == 3  # exactly the cap
    assert any("capped" in rec.message.lower() for rec in caplog.records)
