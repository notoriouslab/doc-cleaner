"""Tests for same-page fragment stitching (stitch-table-fragments).

Pins the "Same-page fragment stitching" requirement: degenerate single-row
table fragments consolidate into the preceding compatible table; multi-row
neighbors, different column counts, oversized gaps/offsets, and intervening
text all break the chain. Constants: _STITCH_X_TOL=6.0, _STITCH_GAP_MAX=30.0.
"""
import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from parsers.pdf import _TablePart, _TextPart, _stitch_page_fragments


def frag(row, y, y1=None, x0=34.5, x1=560.5):
    """A degenerate single-row fragment: its only row sits in header position."""
    return _TablePart(y, header=row, rows=[], x0=x0, x1=x1, y1=y1 if y1 else y + 12.6)


def table(header, rows, y, y1, x0=34.5, x1=560.5):
    return _TablePart(y, header=header, rows=rows, x0=x0, x1=x1, y1=y1)


class TestStitching:
    def test_header_fragment_plus_run_consolidates(self):
        # mirrors the credit-card transaction page: header fragment then rows
        parts = [
            frag(["消費日", "卡號", "金額"], y=355.3, y1=381.3),
            frag(["02/01", "1234", "500"], y=394.3),
            frag(["02/03", "1234", "120"], y=419.5),
            frag(["02/05", "1234", "80"], y=444.8),
        ]
        out = _stitch_page_fragments(parts)
        assert len(out) == 1
        t = out[0]
        assert t.header == ["消費日", "卡號", "金額"]
        assert t.rows == [["02/01", "1234", "500"], ["02/03", "1234", "120"], ["02/05", "1234", "80"]]

    def test_chain_extends_geometry(self):
        # third fragment compares against the stitched result's bottom
        parts = [
            frag(["h1", "h2"], y=100, y1=112),
            frag(["a", "1"], y=130, y1=142),   # gap 18 from first
            frag(["b", "2"], y=160, y1=172),   # gap 18 from stitched bottom (142)
        ]
        out = _stitch_page_fragments(parts)
        assert len(out) == 1
        assert out[0].rows == [["a", "1"], ["b", "2"]]

    def test_multi_row_neighbor_not_stitched(self):
        # the points-table counter-example: same cols, small gap, but rows==3
        parts = [
            table(["信用額度", "循環利率"], [["1000000", "9.9%"]], y=333.6, y1=360.3),
            table(["", ""], [["本期點數", "可用點數"], ["100", "200"], ["x", "y"]], y=374.6, y1=431.8),
        ]
        out = _stitch_page_fragments(parts)
        assert len(out) == 2

    def test_different_column_count_not_stitched_even_at_5pt(self):
        # the securities counter-example: 12 vs 17 cols, 5.5pt gap
        parts = [
            frag(["a"] * 12, y=121, y1=353),
            frag(["b"] * 17, y=358.5),
        ]
        out = _stitch_page_fragments(parts)
        assert len(out) == 2

    def test_intervening_text_breaks_chain(self):
        parts = [
            frag(["h1", "h2"], y=100, y1=112),
            frag(["a", "1"], y=125, y1=137),
            _TextPart(y=150, text="漏網的一列文字"),
            frag(["b", "2"], y=160, y1=172),
            frag(["c", "3"], y=185, y1=197),
        ]
        out = _stitch_page_fragments(parts)
        tables = [p for p in out if isinstance(p, _TablePart)]
        assert len(tables) == 2
        assert tables[0].rows == [["a", "1"]]
        assert tables[1].rows == [["c", "3"]]

    def test_gap_boundary_29_stitches_31_does_not(self):
        base = frag(["h1", "h2"], y=100, y1=112)
        near = frag(["a", "1"], y=141, y1=153)    # gap 29
        far = frag(["a", "1"], y=143.1, y1=155)   # gap 31.1
        assert len(_stitch_page_fragments([base, near])) == 1
        assert len(_stitch_page_fragments([base, far])) == 2

    def test_x_offset_boundary_5_stitches_7_does_not(self):
        base = frag(["h1", "h2"], y=100, y1=112)
        near = frag(["a", "1"], y=125, x0=34.5 + 5, x1=560.5 + 5)
        far = frag(["a", "1"], y=125, x0=34.5 + 7, x1=560.5 + 7)
        assert len(_stitch_page_fragments([base, near])) == 1
        assert len(_stitch_page_fragments([base, far])) == 2

    def test_missing_geometry_never_stitches(self):
        parts = [
            _TablePart(100, header=["h1", "h2"], rows=[]),   # no geometry
            _TablePart(120, header=["a", "1"], rows=[]),
        ]
        assert len(_stitch_page_fragments(parts)) == 2

    def test_pure_no_input_mutation_on_repeat(self):
        parts = [
            frag(["h1", "h2"], y=100, y1=112),
            frag(["a", "1"], y=125, y1=137),
        ]
        first = _stitch_page_fragments(parts)
        second = _stitch_page_fragments(parts)
        assert first[0].rows == [["a", "1"]]
        assert second[0].rows == [["a", "1"]]
        assert parts[0].rows == [] and parts[0].header == ["h1", "h2"]
