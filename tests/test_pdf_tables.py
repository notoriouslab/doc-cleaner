"""Unit + integration tests for PyMuPDF-path table extraction (parsers/pdf.py).

Covers the pdf-table-extraction spec: merged-cell rendering, structural
integrity (pipe escaping, newline collapse, padding), header selection,
cross-page table continuation, interleaving, and fallback behavior.

Integration tests read the committed fixtures in tests/fixtures/ (generated
by make_pdf_table_fixtures.py, which needs reportlab; the tests themselves
only need PyMuPDF).
"""
import os

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")

from parsers.pdf import (
    _TablePart,
    _TextPart,
    _merge_cross_page_tables,
    _table_to_markdown,
    extract_text_with_tables,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name):
    path = os.path.join(FIXTURES, name)
    if not os.path.exists(path):
        pytest.skip(f"fixture {name} missing — run make_pdf_table_fixtures.py")
    return path


# ---------------------------------------------------------------- renderer

class TestTableToMarkdown:
    def test_rowspan_first_cell_filled_rest_blank(self):
        # Exact example from the spec (Requirement: Merged-cell rendering)
        rows = [
            ["項目", "年度", "金額"],
            ["長期趨勢說明", "2023", "100"],
            [None, "2024", "120"],
            [None, "2025", "135"],
        ]
        expected = (
            "|項目|年度|金額|\n"
            "|---|---|---|\n"
            "|長期趨勢說明|2023|100|\n"
            "| |2024|120|\n"
            "| |2025|135|"
        )
        assert _table_to_markdown(rows) == expected

    def test_colspan_section_row(self):
        rows = [
            ["項目", "年度", "金額", "備註"],
            ["經常性支出小計說明", None, None, None],
            ["單項支出", "2025", "42", "一次性"],
        ]
        out = _table_to_markdown(rows)
        assert "|經常性支出小計說明| | | |" in out
        assert out.count("經常性支出小計說明") == 1

    def test_pipe_escaped(self):
        rows = [["A", "B"], ["x|y", "z"]]
        out = _table_to_markdown(rows)
        assert "x\\|y" in out
        # every rendered line keeps the 2-column structure
        for line in out.split("\n"):
            assert line.replace("\\|", "").count("|") == 3

    def test_newlines_collapsed_no_br(self):
        rows = [["A", "B"], ["第一行\n第二行", "v"]]
        out = _table_to_markdown(rows)
        assert "第一行 第二行" in out
        assert "<br>" not in out
        assert "\n第二行" not in out

    def test_ragged_rows_padded(self):
        rows = [["A", "B", "C"], ["1"], ["1", "2"]]
        out = _table_to_markdown(rows)
        for line in out.split("\n"):
            assert line.count("|") == 4  # constant column count

    def test_empty_input(self):
        assert _table_to_markdown([]) == ""
        assert _table_to_markdown([[None, None]]) == ""

    def test_headerless_first_row_promoted_no_coln(self):
        rows = [["甲類", "100"], ["乙類", "200"]]
        out = _table_to_markdown(rows)
        assert out.startswith("|甲類|100|")
        assert "Col1" not in out

    def test_external_header_all_rows_are_data(self):
        rows = [["甲類", "100"], ["乙類", "200"]]
        out = _table_to_markdown(rows, external_header=["類別", "金額"])
        lines = out.split("\n")
        assert lines[0] == "|類別|金額|"
        assert lines[2] == "|甲類|100|"
        assert lines[3] == "|乙類|200|"

    def test_none_in_header_resolves_to_blank(self):
        rows = [["標題", None], ["a", "b"]]
        out = _table_to_markdown(rows)
        assert out.split("\n")[0] == "|標題| |"

    def test_empty_external_header_treated_as_absent(self):
        rows = [["甲類", "100"], ["乙類", "200"]]
        assert _table_to_markdown(rows, external_header=[]) == _table_to_markdown(rows)

    def test_trailing_backslash_cannot_eat_delimiter(self):
        rows = [["path", "val"], ["C:\\", "x"]]
        out = _table_to_markdown(rows)
        assert "|C:\\\\|x|" in out           # backslash escaped, delimiter intact
        for line in out.split("\n"):
            assert line.replace("\\\\", "").replace("\\|", "").count("|") == 3


# ---------------------------------------------------------- cross-page merge

def _tbl(header, rows, y=0.0):
    return _TablePart(y=y, header=header, rows=rows)


class TestMergeCrossPageTables:
    def test_same_header_continuation_merges(self):
        pages = [
            [_tbl(["編號", "金額"], [["001", "10"], ["002", "20"]])],
            [_tbl(["編號", "金額"], [["003", "30"], ["004", "40"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        assert len(merged) == 1
        assert merged[0].rows == [["001", "10"], ["002", "20"], ["003", "30"], ["004", "40"]]

    def test_boundary_duplicate_row_emitted_once(self):
        pages = [
            [_tbl(["編號", "金額"], [["001", "10"], ["002", "20"]])],
            [_tbl(["編號", "金額"], [["002", "20"], ["003", "30"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        assert merged[0].rows == [["001", "10"], ["002", "20"], ["003", "30"]]

    def test_different_headers_stay_separate(self):
        pages = [
            [_tbl(["編號", "金額"], [["001", "10"]])],
            [_tbl(["科目", "說明"], [["奉獻", "x"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        assert len(merged) == 2
        assert merged[1].rows == [["奉獻", "x"]]

    def test_same_text_different_column_count_not_merged(self):
        pages = [
            [_tbl(["編號", "金額"], [["001", "10"]])],
            [_tbl(["編號", "金額", "備註"], [["002", "20", "n"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        assert len(merged) == 2

    def test_three_page_continuation(self):
        pages = [
            [_tbl(["h"], [["1"]])],
            [_tbl(["h"], [["2"]])],
            [_tbl(["h"], [["3"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        assert len(merged) == 1
        assert merged[0].rows == [["1"], ["2"], ["3"]]

    def test_text_only_page_breaks_continuation(self):
        pages = [
            [_tbl(["h"], [["1"]])],
            [_TextPart(y=0.0, text="純文字頁")],
            [_tbl(["h"], [["2"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        tables = [p for p in merged if isinstance(p, _TablePart)]
        assert len(tables) == 2

    def test_only_first_table_of_page_merges(self):
        # second table of page 2 has same header but is not a continuation
        pages = [
            [_tbl(["h"], [["1"]])],
            [_tbl(["h"], [["2"]], y=10.0), _tbl(["h"], [["9"]], y=500.0)],
        ]
        merged = _merge_cross_page_tables(pages)
        tables = [p for p in merged if isinstance(p, _TablePart)]
        assert len(tables) == 2
        assert tables[0].rows == [["1"], ["2"]]
        assert tables[1].rows == [["9"]]

    def test_identical_full_table_on_both_pages_emitted_once(self):
        # e.g. the same table fully re-detected on two consecutive pages
        pages = [
            [_tbl(["h", "v"], [["a", "1"], ["b", "2"]])],
            [_tbl(["h", "v"], [["a", "1"], ["b", "2"]])],
        ]
        merged = _merge_cross_page_tables(pages)
        assert len(merged) == 1
        assert merged[0].rows == [["a", "1"], ["b", "2"]]

    def test_pure_no_input_mutation_on_repeat_call(self):
        pages = [
            [_tbl(["h"], [["1"]])],
            [_tbl(["h"], [["2"]])],
        ]
        first = _merge_cross_page_tables(pages)
        second = _merge_cross_page_tables(pages)
        assert first[0].rows == [["1"], ["2"]]
        assert second[0].rows == [["1"], ["2"]]          # no doubling
        assert pages[0][0].rows == [["1"]]               # input untouched

    def test_text_parts_preserved_in_order(self):
        pages = [
            [_TextPart(y=0.0, text="前言"), _tbl(["h"], [["1"]], y=50.0)],
            [_tbl(["h"], [["2"]], y=10.0), _TextPart(y=100.0, text="結語")],
        ]
        merged = _merge_cross_page_tables(pages)
        kinds = ["text" if isinstance(p, _TextPart) else "table" for p in merged]
        assert kinds == ["text", "table", "text"]
        assert merged[1].rows == [["1"], ["2"]]


# ------------------------------------------------------------- integration

class TestFixtureIntegration:
    def test_merged_cells_blank_not_repeated(self):
        out = extract_text_with_tables(fixture("merged_cells.pdf"))
        assert out is not None
        # row-span title appears exactly once (was 3× before the fix)
        assert out.count("長期趨勢與成效評估說明") == 1
        # column-span section text appears exactly once (was 4× before)
        assert out.count("經常性支出小計說明") == 1
        # data rows of the spanned block are intact
        for cell in ("2023", "2024", "2025"):
            assert cell in out

    def test_crosspage_no_data_loss_single_header(self):
        out = extract_text_with_tables(fixture("crosspage_repeated_header.pdf"))
        assert out is not None
        rows = [ln for ln in out.split("\n") if ln.startswith("|")]
        data_rows = [ln for ln in rows if ln.split("|")[1].strip().isdigit()]
        assert len(data_rows) == 60          # was 37 before the fix
        assert "060" in out
        header_rows = [ln for ln in rows if "編號" in ln]
        assert len(header_rows) == 1         # merged: single header

    def test_special_chars_structure_intact(self):
        out = extract_text_with_tables(fixture("special_chars.pdf"))
        assert out is not None
        table_lines = [ln for ln in out.split("\n") if ln.startswith("|")]
        counts = {ln.replace("\\|", "").count("|") for ln in table_lines}
        assert counts == {3}                 # constant 2-column structure
        assert "a\\|b" in out

    def test_no_header_no_coln_placeholder(self):
        out = extract_text_with_tables(fixture("no_header.pdf"))
        assert out is not None
        assert "Col1" not in out and "Col2" not in out
        for cell in ("甲類", "乙類", "丙類"):
            assert cell in out

    def test_mixed_content_order_preserved(self):
        out = extract_text_with_tables(fixture("mixed_content.pdf"))
        assert out is not None
        i_above = out.index("表格上方的說明段落")
        i_table = out.index("奉獻收入")
        i_below = out.index("表格下方的結論段落")
        assert i_above < i_table < i_below

    def test_document_level_error_returns_none(self):
        assert extract_text_with_tables("/nonexistent/no_such.pdf") is None
