"""Tests for parsers/iwork.py — multi-strategy Apple Pages/Keynote extraction.

Coverage:
- QuickLook-PDF path (shared helper): real ZIP fixtures with a PyMuPDF-generated PDF.
- IWA path: pure-logic unit tests on the traversal/ordering helpers (no real .key
  needed), plus an end-to-end check against a real .key when one is available.
- Dispatch, fallback, and the .pages degrade message.
"""
import logging
import os
import tempfile
import zipfile
from unittest.mock import patch

import pytest

from parsers import iwork


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_iwork_file(tmp_path, ext, entries):
    """Build a real ZIP at tmp_path/doc<ext>. entries: {arcname: bytes}."""
    p = tmp_path / f"doc{ext}"
    with zipfile.ZipFile(p, "w") as zf:
        for arcname, data in entries.items():
            zf.writestr(arcname, data)
    return str(p)


def _minimal_pdf_bytes(text="Hello iWork QuickLook"):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _spy_named_temp():
    created = []
    orig = tempfile.NamedTemporaryFile

    def spy(*args, **kwargs):
        f = orig(*args, **kwargs)
        created.append(f.name)
        return f

    return created, patch("parsers.iwork.tempfile.NamedTemporaryFile", side_effect=spy)


REAL_KEY = "/Users/jacobmei/Desktop/整理/2026-06-18 AI 時代的職涯規劃.key"


# ── TestPages: QuickLook-PDF primary, else degrade ─────────────────────────────

class TestPages:
    def test_pages_success(self, tmp_path):
        """A .pages with a QuickLook PDF extracts via the PDF pipeline."""
        path = _make_iwork_file(
            tmp_path, ".pages",
            {"QuickLook/Preview.pdf": _minimal_pdf_bytes("Pages body text")},
        )
        assert "Pages body text" in iwork.parse(path)

    def test_pages_modern_degrade(self, tmp_path, caplog):
        """A modern .pages (no QuickLook PDF) returns '' with an export instruction."""
        path = _make_iwork_file(tmp_path, ".pages", {"Index/Document.iwa": b"x"})
        with caplog.at_level(logging.WARNING):
            result = iwork.parse(path)
        assert result == ""
        assert "doc.pages" in caplog.text
        assert "export" in caplog.text.lower()


# ── TestKeynoteFallback: IWA-empty falls back to QuickLook PDF ──────────────────

class TestKeynoteFallback:
    def test_key_falls_back_to_quicklook_pdf(self, tmp_path):
        """A .key with no IWA slide text but a QuickLook PDF extracts via fallback."""
        path = _make_iwork_file(
            tmp_path, ".key",
            {"QuickLook/Preview.pdf": _minimal_pdf_bytes("Legacy keynote text")},
        )
        assert "Legacy keynote text" in iwork.parse(path)

    def test_key_no_iwa_no_pdf_returns_empty(self, tmp_path, caplog):
        """A .key with neither IWA text nor a QuickLook PDF returns '' + warning."""
        path = _make_iwork_file(tmp_path, ".key", {"Index/Document.iwa": b"x"})
        with caplog.at_level(logging.WARNING):
            result = iwork.parse(path)
        assert result == ""
        assert "doc.key" in caplog.text


# ── TestIWATraversal: pure-logic helpers (no real .key needed) ──────────────────

class TestIWATraversal:
    def test_collect_text_strips_placeholder_and_blanks(self):
        tree = {"objects": [{"text": ["Hello", "￼", "  ", "World￼"]}]}
        hits = []
        iwork._collect_text(tree, hits)
        assert hits == ["Hello", "World"]

    def test_collect_text_recurses_nested(self):
        tree = {"a": [{"b": {"text": ["deep"]}}], "text": ["top"]}
        hits = []
        iwork._collect_text(tree, hits)
        assert set(hits) == {"deep", "top"}

    def test_find_first_key(self):
        obj = {"x": [1, {"slideTree": {"slides": [{"identifier": "9"}]}}]}
        tree = iwork._find_first_key(obj, "slideTree")
        assert tree == {"slides": [{"identifier": "9"}]}

    def test_find_first_key_absent(self):
        assert iwork._find_first_key({"a": 1}, "missing") is None

    def test_slide_file_id(self):
        assert iwork._slide_file_id("Index/Slide-1595612.iwa") == "1595612"
        assert iwork._slide_file_id("Index/Document.iwa") is None
        assert iwork._slide_file_id("Index/Slide-Master.iwa") is None

    def test_proxy_to_file_id(self):
        store = {
            "1686744": {"header": {"messageInfos": [
                {"objectReferences": ["1686741"]}]}},
        }
        assert iwork._proxy_to_file_id(store, "1686744") == "1686741"
        assert iwork._proxy_to_file_id(store, "missing") is None

    def test_proxy_to_file_id_no_refs(self):
        store = {"1": {"header": {"messageInfos": [{}]}}}
        assert iwork._proxy_to_file_id(store, "1") is None


# ── TestKeynoteZipBomb: IWA-path pre-decode size guard (D9) ─────────────────────

class TestKeynoteZipBomb:
    def test_within_limit_true(self, tmp_path):
        path = _make_iwork_file(tmp_path, ".key", {"Index/Document.iwa": b"small"})
        assert iwork._within_size_limit(path, "doc.key") is True

    def test_over_limit_skips_iwa(self, tmp_path, caplog):
        path = _make_iwork_file(tmp_path, ".key", {"Index/Document.iwa": b"x" * 50})
        with patch.object(iwork, "MAX_DECOMPRESSED_SIZE", 10):
            with caplog.at_level(logging.WARNING):
                ok = iwork._within_size_limit(path, "doc.key")
        assert ok is False
        assert "skipping IWA" in caplog.text

    def test_parse_key_over_limit_does_not_decode_iwa(self, tmp_path):
        """When the size guard trips, the IWA decoder is never invoked."""
        path = _make_iwork_file(
            tmp_path, ".key",
            {"Index/Document.iwa": b"x" * 50,
             "QuickLook/Preview.pdf": _minimal_pdf_bytes("fallback text")},
        )
        with patch.object(iwork, "MAX_DECOMPRESSED_SIZE", 10), \
             patch("parsers.iwork._extract_iwa_slides") as mock_iwa:
            # guard trips before IWA; fallback PDF path has its own guard, so it
            # also rejects the oversize entry — result is "" but IWA never ran.
            iwork.parse(path)
        mock_iwa.assert_not_called()


# ── TestDispatch ───────────────────────────────────────────────────────────────

class TestDispatch:
    def test_key_routes_to_parse_key(self, tmp_path):
        path = _make_iwork_file(tmp_path, ".key", {"Index/Document.iwa": b"x"})
        with patch("parsers.iwork._parse_key", return_value="KEY") as m:
            assert iwork.parse(path) == "KEY"
            m.assert_called_once()

    def test_pages_routes_to_parse_pages(self, tmp_path):
        path = _make_iwork_file(tmp_path, ".pages", {"Index/Document.iwa": b"x"})
        with patch("parsers.iwork._parse_pages", return_value="PAGES") as m:
            assert iwork.parse(path) == "PAGES"
            m.assert_called_once()


# ── TestKeynoteMissingLibrary ──────────────────────────────────────────────────

class TestKeynoteMissingLibrary:
    def test_missing_keynote_parser_yields_no_iwa(self, tmp_path, caplog):
        """If keynote-parser can't import, IWA yields '' so fallback can run."""
        path = _make_iwork_file(tmp_path, ".key", {"Index/Document.iwa": b"x"})
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name == "keynote_parser.codec" or name.startswith("keynote_parser"):
                raise ImportError("simulated missing keynote_parser")
            return real_import(name, *a, **k)

        with patch("builtins.__import__", side_effect=fake_import):
            with caplog.at_level(logging.WARNING):
                result = iwork._extract_iwa_slides(path, "doc.key")
        assert result == ""
        assert "keynote-parser not installed" in caplog.text


# ── TestQuickLookPdfPath: shared helper (temp cleanup, guard, corrupt) ──────────

class TestQuickLookPdfPath:
    def test_temp_deleted_after_success(self, tmp_path):
        path = _make_iwork_file(
            tmp_path, ".pages", {"QuickLook/Preview.pdf": _minimal_pdf_bytes()})
        created, spy = _spy_named_temp()
        with spy:
            iwork.parse(path)
        assert created and all(not os.path.exists(p) for p in created)

    def test_temp_deleted_after_pipeline_failure(self, tmp_path):
        path = _make_iwork_file(
            tmp_path, ".pages", {"QuickLook/Preview.pdf": b"%PDF-1.4 not a real pdf"})
        created, spy = _spy_named_temp()
        with spy:
            result = iwork.parse(path)
        assert result == ""
        assert created and all(not os.path.exists(p) for p in created)

    def test_quicklook_over_limit_rejected(self, tmp_path, caplog):
        path = _make_iwork_file(
            tmp_path, ".pages", {"QuickLook/Preview.pdf": b"small"})
        fake = zipfile.ZipInfo(iwork.QUICKLOOK_PDF)
        fake.file_size = iwork.MAX_DECOMPRESSED_SIZE + 1
        with patch.object(zipfile.ZipFile, "getinfo", return_value=fake), \
             patch.object(zipfile.ZipFile, "read") as mock_read:
            with caplog.at_level(logging.WARNING):
                result = iwork.parse(path)
        assert result == ""
        mock_read.assert_not_called()
        assert "too large" in caplog.text

    def test_corrupt_pdf_bytes(self, tmp_path, caplog):
        path = _make_iwork_file(
            tmp_path, ".pages",
            {"QuickLook/Preview.pdf": b"%PDF-1.4\ngarbage that fitz cannot open"})
        with caplog.at_level(logging.WARNING):
            result = iwork.parse(path)
        assert result == ""
        assert "doc.pages" in caplog.text


# ── TestRealKeynote: end-to-end against a real .key (skipped if absent) ─────────

@pytest.mark.skipif(not os.path.exists(REAL_KEY), reason="real .key fixture absent")
class TestRealKeynote:
    def test_extracts_ordered_chinese_slides(self):
        """Full IWA chain: real modern .key yields ordered `## 投影片 N` Chinese text.

        Ordering ground truth (recorded by opening the deck in Keynote): the
        cover slide (投影片 1) contains the presentation title "AI 時代的職涯規劃".
        """
        result = iwork.parse(REAL_KEY)
        assert result.count("## 投影片") >= 50
        assert "AI 時代的職涯規劃" in result
        assert "街口支付" in result
        # cover title appears in the first slide section, proving slideTree order
        first_section = result.split("## 投影片")[1]
        assert "AI 時代的職涯規劃" in first_section

    def test_no_temp_files_left(self):
        created, spy = _spy_named_temp()
        with spy:
            iwork.parse(REAL_KEY)
        # IWA path uses no temp file; if fallback ran, temps must be cleaned
        assert all(not os.path.exists(p) for p in created)


# ── TestProtobufCoexistence: D10 guard against descriptor-pool crash ────────────

class TestProtobufCoexistence:
    def test_pure_python_protobuf_guard_set(self):
        """Importing parsers.iwork sets the protobuf implementation guard (D10)."""
        assert os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python"

    def test_numbers_and_keynote_coexist_in_one_process(self):
        """Both IWA libraries import together without a descriptor-pool crash.

        Under the C/upb implementation this raises 'duplicate file name
        TSDArchives.proto'; the D10 guard selects the pure-Python implementation,
        which tolerates the shared Apple .proto names.
        """
        pytest.importorskip("numbers_parser")
        pytest.importorskip("keynote_parser")
        import numbers_parser  # noqa: F401
        import keynote_parser.codec  # noqa: F401
        from google.protobuf.internal import api_implementation
        assert api_implementation.Type() == "python"
