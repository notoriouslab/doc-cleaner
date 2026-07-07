"""Tests for collapsed-row column rebuild (rebuild-collapsed-rows).

Pure core: _bucket_words_into_columns(words, boundaries) — word x-centers
bucket into boundary ranges; always returns len(boundaries) strings.
Adapter: collapsed rows inside a detected table regain their columns using
the table's own boundary row; full-width titles stay first-cell.
"""
import os

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from parsers.pdf import _bucket_words_into_columns

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def w(x0, text, y0=0.0, width=10.0):
    """Word tuple in page.get_text('words') shape (extra fields tolerated)."""
    return (x0, y0, x0 + width, y0 + 10.0, text, 0, 0, 0)


class TestBucketWords:
    BOUNDS_3 = [(0, 100), (100, 200), (200, 300)]

    def test_stock_row_shape(self):
        # mirrors the 12-column securities row (condensed to 3 columns)
        words = [w(10, "現股"), w(120, "2330"), w(150, "台積電"), w(250, "580,000")]
        out = _bucket_words_into_columns(words, self.BOUNDS_3)
        assert out == ["現股", "2330 台積電", "580,000"]

    def test_wrapped_continuation_lands_in_its_column(self):
        words = [w(10, "現股", y0=0), w(120, "元大台灣高息", y0=0), w(250, "9.99%", y0=0),
                 w(120, "低波", y0=12)]   # wrapped second line, same x-range
        out = _bucket_words_into_columns(words, self.BOUNDS_3)
        assert out[1] == "元大台灣高息 低波"

    def test_right_aligned_value_at_boundary_edge(self):
        words = [w(85, "999,999", width=14)]   # center 92 → column 0
        out = _bucket_words_into_columns(words, self.BOUNDS_3)
        assert out == ["999,999", "", ""]

    def test_out_of_range_attaches_to_nearest(self):
        words = [w(320, "溢出", width=10)]     # center 325, beyond all → nearest is col 2
        out = _bucket_words_into_columns(words, self.BOUNDS_3)
        assert out == ["", "", "溢出"]

    def test_empty_words_all_empty(self):
        assert _bucket_words_into_columns([], self.BOUNDS_3) == ["", "", ""]

    def test_single_boundary_collects_everything(self):
        words = [w(10, "a"), w(500, "b")]
        assert _bucket_words_into_columns(words, [(0, 100)]) == ["a b"]

    def test_narrow_table_straddling_title_documented_split(self):
        # accepted behavior: a title straddling 2 columns renders split
        words = [w(80, "季度", width=30), w(120, "小計", width=30)]
        out = _bucket_words_into_columns(words, [(0, 100), (100, 200)])
        assert out == ["季度", "小計"]

    def test_reading_order_sorted_by_y_then_x(self):
        words = [w(10, "後", y0=20), w(10, "前", y0=0)]
        out = _bucket_words_into_columns(words, [(0, 100)])
        assert out == ["前 後"]


class TestAdapterFixture:
    def test_header_grid_only_fixture_rebuilds(self):
        from parsers.pdf import extract_text_with_tables
        path = os.path.join(FIXTURES, "header_grid_only.pdf")
        if not os.path.exists(path):
            pytest.skip("fixture missing — run make_pdf_table_fixtures.py")
        out = extract_text_with_tables(path)
        assert out is not None
        lines = [ln for ln in out.split("\n") if ln.startswith("|")]
        # body rows regained their columns
        assert any(ln.startswith("|現股|2330|台積電|") for ln in lines)
        assert any(ln.startswith("|現股|0050|元大台灣50|") for ln in lines)
        # full-width title stays first-cell (single non-empty cell)
        title = [ln for ln in lines if "小計標題列跨全寬置中" in ln]
        assert title and all(
            sum(1 for c in ln.split("|")[1:-1] if c.strip()) == 1 for ln in title)
        # constant column count across the table
        widths = {ln.replace("\\|", "").count("|") for ln in lines}
        assert widths == {6}
