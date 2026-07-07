"""Unit tests for parsers/_tableutil.py — shared table-cell normalization/escaping.

Pins the cross-format contract from the table-cell-escaping spec: None → "",
str() for non-strings, whitespace runs collapsed via re.sub(r'\\s+', ' '),
backslash escaped before pipe.
"""
from parsers._tableutil import escape_cell, normalize_cell


class TestNormalizeCell:
    def test_none_is_empty(self):
        assert normalize_cell(None) == ""

    def test_int_and_float(self):
        assert normalize_cell(3) == "3"
        assert normalize_cell(1.5) == "1.5"

    def test_newline_collapsed(self):
        assert normalize_cell("第一行\n第二行") == "第一行 第二行"

    def test_tab_and_multi_space_collapsed(self):
        assert normalize_cell("a\t b  \n c") == "a b c"

    def test_stripped(self):
        assert normalize_cell("  x  ") == "x"

    def test_no_escaping(self):
        assert normalize_cell("a|b") == "a|b"
        assert normalize_cell("C:\\") == "C:\\"


class TestEscapeCell:
    def test_pipe_in_text(self):
        assert escape_cell("a|b") == "a\\|b"

    def test_trailing_backslash(self):
        # 'C:\' → 'C:\\' (two characters backslash backslash)
        assert escape_cell("C:\\") == "C:\\\\"

    def test_preescaped_pipe_in_source_text(self):
        # source text literally contains backslash-pipe: both chars escaped
        assert escape_cell("a\\|b") == "a\\\\\\|b"

    def test_newline_collapsed(self):
        assert escape_cell("第一行\n第二行") == "第一行 第二行"

    def test_none_is_empty(self):
        assert escape_cell(None) == ""

    def test_float_passthrough(self):
        assert escape_cell(1.5) == "1.5"

    def test_combined(self):
        assert escape_cell("  x|y\\ \n z ") == "x\\|y\\\\ z"

    def test_never_raises(self):
        for v in (None, 0, 0.0, "", [], {}, object()):
            escape_cell(v)  # must not raise
