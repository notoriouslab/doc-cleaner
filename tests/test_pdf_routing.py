"""Tests for first-party table-scan routing (route-pdf-tables-native).

Pins the modified "ODL text extraction" requirement: table-bearing PDFs skip
ODL and take the native PyMuPDF table path; no-table documents keep the
ODL-first flow. Plus clean_odl_output <br> stripping.
"""
import os

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from parsers.pdf import clean_odl_output, has_tables

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class TestHasTables:
    def test_true_on_table_fixture(self):
        assert has_tables(os.path.join(FIXTURES, "merged_cells.pdf")) is True

    def test_false_on_text_only_pdf(self, tmp_path):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "plain prose only, no tables here")
        p = str(tmp_path / "prose.pdf")
        doc.save(p)
        doc.close()
        assert has_tables(p) is False

    def test_false_on_nonexistent_path(self):
        assert has_tables("/nonexistent/nope.pdf") is False

    def test_false_on_corrupted_pdf(self, tmp_path):
        p = tmp_path / "broken.pdf"
        p.write_bytes(b"%PDF-1.4 truncated garbage \x00\x01")
        assert has_tables(str(p)) is False

    def test_false_on_injected_exception(self, monkeypatch):
        import parsers.pdf as pdfmod

        def boom(*a, **k):
            raise RuntimeError("injected")
        monkeypatch.setattr(pdfmod.fitz, "open", boom)
        assert has_tables(os.path.join(FIXTURES, "merged_cells.pdf")) is False


class TestRouting:
    def _convert(self, src, tmp_path, monkeypatch, sentinel):
        from cleaner import load_config, process_file
        import parsers.pdf as pdfmod
        monkeypatch.setattr(pdfmod, "extract_text_odl", sentinel)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir, exist_ok=True)
        config = load_config(None)
        return process_file(src, None, None, config, out_dir)

    def test_table_pdf_skips_odl(self, tmp_path, monkeypatch):
        calls = []

        def sentinel(path):
            calls.append(path)
            return None
        status, out_path = self._convert(
            os.path.join(FIXTURES, "merged_cells.pdf"), tmp_path, monkeypatch, sentinel)
        assert calls == []                      # ODL was never invoked
        assert status == "ok"
        with open(out_path, encoding="utf-8") as f:
            text = f.read()
        assert "|項目|年度|金額|備註|" in text  # native table path output
        assert "<br>" not in text

    def test_no_table_pdf_calls_odl(self, tmp_path, monkeypatch):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "plain prose document without any table structure")
        src = str(tmp_path / "prose.pdf")
        doc.save(src)
        doc.close()
        calls = []

        def sentinel(path):
            calls.append(path)
            return None                          # behave as ODL-unavailable
        status, _ = self._convert(src, tmp_path, monkeypatch, sentinel)
        assert len(calls) == 1                   # ODL-first flow preserved


class TestCleanOdlBr:
    def test_all_br_variants_stripped(self):
        s = "a<br>b<br/>c<br />d"
        out = clean_odl_output(s)
        assert "<br" not in out
        assert out == "a b c d"

    def test_mixed_with_tables(self):
        s = "|欄|值|\n|---|---|\n|x<br><br>y|z|"
        out = clean_odl_output(s)
        assert "<br" not in out
        assert "|x  y|z|" in out or "|x y|z|" in out
