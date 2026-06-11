"""Tests for parsers/numbers.py — all cases use mocked numbers_parser.Document."""
import logging
import sys
from unittest.mock import MagicMock, patch


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cell(value):
    c = MagicMock()
    c.value = value
    return c


def _make_row(*values):
    row = MagicMock()
    row.__iter__ = MagicMock(return_value=iter([_make_cell(v) for v in values]))
    return row


def _make_table(name, rows_data):
    """rows_data: list of tuples/lists of values; first row = header."""
    t = MagicMock()
    t.name = name
    mock_rows = [_make_row(*r) for r in rows_data]
    t.rows = MagicMock(return_value=mock_rows)
    return t


def _make_sheet(name, tables):
    s = MagicMock()
    s.name = name
    s.tables = tables
    return s


def _make_doc(sheets):
    d = MagicMock()
    d.sheets = sheets
    return d


# ── TestExtraction ────────────────────────────────────────────────────────────

class TestExtraction:
    def test_single_sheet_single_table(self):
        """Extract all sheets as Markdown pipe tables — single sheet, single table."""
        from parsers.numbers import parse

        table = _make_table("Table 1", [
            ("Name", "Age"),
            ("Alice", 30),
            ("Bob", 25),
        ])
        doc = _make_doc([_make_sheet("Summary", [table])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert result.startswith("## Summary\n\n")
        assert "| Name | Age |" in result
        assert "| Alice | 30 |" in result
        assert "| Bob | 25 |" in result

    def test_multi_sheet_in_order(self):
        """Multi-sheet extraction — sections appear in document order."""
        from parsers.numbers import parse

        t1 = _make_table("T", [("Q", "V"), ("q1", 1)])
        t2 = _make_table("T", [("Q", "V"), ("q2", 2)])
        doc = _make_doc([_make_sheet("Q1", [t1]), _make_sheet("Q2", [t2])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        q1_pos = result.index("## Q1")
        q2_pos = result.index("## Q2")
        assert q1_pos < q2_pos
        assert "| q1 |" in result
        assert "| q2 |" in result

    def test_multi_table_sheet_heading(self):
        """Multi-table sheet uses '## SheetName — TableName' heading."""
        from parsers.numbers import parse

        t_budget = _make_table("Budget", [("Item",), ("Rent",)])
        t_actual = _make_table("Actuals", [("Item",), ("Rent",)])
        doc = _make_doc([_make_sheet("Finance", [t_budget, t_actual])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert "## Finance — Budget" in result
        assert "## Finance — Actuals" in result

    def test_single_table_uses_simple_heading(self):
        """Single table per sheet uses '## SheetName' (no table name suffix)."""
        from parsers.numbers import parse

        table = _make_table("Table 1", [("X",), ("1",)])
        doc = _make_doc([_make_sheet("Data", [table])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert "## Data\n\n" in result
        assert "— Table 1" not in result

    def test_empty_sheet_skipped(self):
        """Empty sheet produces no output."""
        from parsers.numbers import parse

        empty_table = _make_table("Empty", [])
        data_table = _make_table("T", [("Col",), ("val",)])
        doc = _make_doc([
            _make_sheet("Empty", [empty_table]),
            _make_sheet("Data", [data_table]),
        ])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert "## Empty" not in result
        assert "## Data" in result

    def test_none_cell_becomes_empty_string(self):
        """None cell values render as empty strings in the table."""
        from parsers.numbers import parse

        table = _make_table("T", [("A", "B"), (None, "x")])
        doc = _make_doc([_make_sheet("S", [table])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert "|  | x |" in result


# ── TestTruncation ────────────────────────────────────────────────────────────

class TestTruncation:
    def test_within_budget_no_truncation(self):
        """Table within 8000 chars has no truncation note."""
        from parsers.numbers import parse

        table = _make_table("T", [("Col",), ("val",)])
        doc = _make_doc([_make_sheet("S", [table])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert "(truncated" not in result

    def test_exceeds_budget_adds_note(self):
        """Table exceeding budget gets truncation note with correct counts."""
        from parsers.numbers import parse

        # Build a table with enough rows to exceed 8000 chars
        header = ("Col1", "Col2", "Col3", "Col4", "Col5")
        data_row = ("A" * 30, "B" * 30, "C" * 30, "D" * 30, "E" * 30)
        rows = [header] + [data_row] * 200  # 200 data rows — will exceed budget

        table = _make_table("T", rows)
        doc = _make_doc([_make_sheet("S", [table])])

        with patch("numbers_parser.Document", return_value=doc):
            result = parse("dummy.numbers")

        assert "(truncated —" in result
        assert "rows shown of 200)" in result

    def test_truncation_note_format(self):
        """Truncation note matches exact format: '(truncated — N rows shown of M)'."""
        from parsers.numbers import _table_to_markdown

        header = tuple(f"C{i}" for i in range(10))
        long_row = tuple("X" * 100 for _ in range(10))
        rows = [header] + [long_row] * 500

        md, total, shown = _table_to_markdown(rows, max_chars=8000)

        assert total == 500
        assert shown < total
        assert f"(truncated — {shown} rows shown of {total})" in md


# ── TestGracefulFailure ───────────────────────────────────────────────────────

class TestGracefulFailure:
    def test_missing_library_returns_empty(self, caplog):
        """Graceful failure when numbers-parser is absent — returns '' without raising."""
        # Remove numbers_parser from sys.modules if present, then make import fail
        sys.modules.pop("numbers_parser", None)

        with patch.dict(sys.modules, {"numbers_parser": None}):
            # Reload the module to reset the cached import state
            import importlib
            import parsers.numbers as nm
            importlib.reload(nm)

            with caplog.at_level(logging.WARNING, logger="parsers.numbers"):
                result = nm.parse("dummy.numbers")

        assert result == ""
        assert "numbers-parser not installed" in caplog.text

    def test_corrupt_file_returns_empty(self, caplog):
        """Graceful failure on corrupt or unreadable file — returns '' without raising."""
        from parsers.numbers import parse

        with patch("numbers_parser.Document", side_effect=Exception("bad zip")):
            with caplog.at_level(logging.WARNING, logger="parsers.numbers"):
                result = parse("corrupt.numbers")

        assert result == ""
        assert "bad zip" in caplog.text

    def test_corrupt_file_no_exception_propagates(self):
        """No exception propagates to the caller for corrupt files."""
        from parsers.numbers import parse

        with patch("numbers_parser.Document", side_effect=RuntimeError("unexpected")):
            result = parse("bad.numbers")  # must not raise

        assert result == ""
