"""Tests for Group 2 parser fixes: absolute textutil path + .xls xlrd engine."""
import os
import platform
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Task 2.1: /usr/bin/textutil absolute path ─────────────────────────────────

@pytest.mark.skipif(platform.system() != "Darwin", reason="textutil is macOS-only")
class TestTextutilAbsolutePath:
    def _make_run_side_effect(self, tmp_path, content="extracted text"):
        """Return a side-effect that writes a fake output file."""
        def _run(cmd, **kwargs):
            idx = list(cmd).index("-output")
            Path(cmd[idx + 1]).write_text(content, encoding="utf-8")
            return MagicMock(returncode=0)
        return _run

    def test_uses_absolute_path(self, tmp_path):
        """subprocess.run is called with /usr/bin/textutil, not bare 'textutil'."""
        from parsers._textutil import convert_to_text

        dummy = str(tmp_path / "test.doc")
        Path(dummy).write_bytes(b"dummy")

        with patch("parsers._textutil.subprocess.run",
                   side_effect=self._make_run_side_effect(tmp_path)) as mock_run:
            convert_to_text(dummy, "DOC")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/textutil", (
            f"Expected /usr/bin/textutil, got {cmd[0]!r}"
        )

    def test_doc_succeeds_with_minimal_path(self, tmp_path):
        """Conversion succeeds even when PATH contains only /usr/bin."""
        from parsers._textutil import convert_to_text

        dummy = str(tmp_path / "test.doc")
        Path(dummy).write_bytes(b"dummy")

        minimal_env = {**os.environ, "PATH": "/usr/bin"}
        with patch("parsers._textutil.subprocess.run",
                   side_effect=self._make_run_side_effect(tmp_path, "doc content")) as mock_run:
            result = convert_to_text(dummy, "DOC")

        assert result == "doc content"
        assert mock_run.called

    def test_ppt_succeeds_with_minimal_path(self, tmp_path):
        """PPT conversion uses the same absolute path and succeeds."""
        from parsers._textutil import convert_to_text

        dummy = str(tmp_path / "test.ppt")
        Path(dummy).write_bytes(b"dummy")

        with patch("parsers._textutil.subprocess.run",
                   side_effect=self._make_run_side_effect(tmp_path, "ppt content")) as mock_run:
            result = convert_to_text(dummy, "PPT")

        assert result == "ppt content"
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/textutil"


# ── Task 2.2: .xls uses xlrd engine explicitly ────────────────────────────────

class TestXlsXlrdEngine:
    def test_xls_passes_xlrd_engine(self, tmp_path):
        """pd.read_excel is called with engine='xlrd' for .xls files."""
        import pandas as pd
        from parsers.xlsx import parse

        dummy_xls = str(tmp_path / "test.xls")
        Path(dummy_xls).write_bytes(b"dummy")

        fake_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with patch("pandas.read_excel", return_value={"Sheet1": fake_df}) as mock_read:
            result = parse(dummy_xls)

        _, kwargs = mock_read.call_args
        assert kwargs.get("engine") == "xlrd", (
            f"Expected engine='xlrd', got {kwargs.get('engine')!r}"
        )

    def test_xlsx_does_not_force_xlrd_engine(self, tmp_path):
        """pd.read_excel for .xlsx does NOT force engine='xlrd' (uses default)."""
        import pandas as pd
        from parsers.xlsx import parse

        dummy_xlsx = str(tmp_path / "test.xlsx")
        Path(dummy_xlsx).write_bytes(b"dummy")

        fake_df = pd.DataFrame({"A": [1]})
        with patch("pandas.read_excel", return_value={"Sheet1": fake_df}) as mock_read:
            parse(dummy_xlsx)

        _, kwargs = mock_read.call_args
        assert kwargs.get("engine") is None, (
            f"xlsx should not set engine, got {kwargs.get('engine')!r}"
        )

    def test_xls_produces_markdown_table(self, tmp_path):
        """End-to-end: .xls path returns Markdown pipe-table content."""
        import pandas as pd
        from parsers.xlsx import parse

        dummy_xls = str(tmp_path / "data.xls")
        Path(dummy_xls).write_bytes(b"dummy")

        fake_df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
        with patch("pandas.read_excel", return_value={"Results": fake_df}):
            result = parse(dummy_xls)

        assert "## Sheet: Results" in result
        assert "Alice" in result
        assert "|" in result  # pipe table format
