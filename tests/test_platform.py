"""Tests for parsers/_platform.py — cross-platform conversion and reveal."""
import platform
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── convert_legacy_office ─────────────────────────────────────────────────────

@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
class TestConvertLegacyOfficeDarwin:
    def test_delegates_to_textutil(self, tmp_path):
        """On macOS, convert_legacy_office calls _textutil.convert_to_text."""
        from parsers._platform import convert_legacy_office

        with patch("parsers._textutil.subprocess.run") as mock_run:
            def _fake_run(cmd, **kw):
                Path(cmd[cmd.index("-output") + 1]).write_text("hello", encoding="utf-8")
                return MagicMock(returncode=0)
            mock_run.side_effect = _fake_run

            dummy = str(tmp_path / "test.doc")
            Path(dummy).write_bytes(b"dummy")
            result = convert_legacy_office(dummy, format_label="DOC")

        assert result == "hello"
        assert mock_run.call_args[0][0][0] == "/usr/bin/textutil"


class TestConvertLegacyOfficeNoLibreOffice:
    def test_returns_empty_and_warns_when_libreoffice_missing(self, tmp_path, caplog):
        """When LibreOffice is not found, returns '' and logs a warning."""
        import logging
        from parsers import _platform

        dummy = str(tmp_path / "test.doc")
        Path(dummy).write_bytes(b"dummy")

        with patch.object(_platform, "SYSTEM", "Windows"), \
             patch("parsers._platform._find_libreoffice", return_value=None), \
             caplog.at_level(logging.WARNING, logger="parsers._platform"):
            result = _platform.convert_legacy_office(dummy, format_label="DOC")

        assert result == ""
        assert any("LibreOffice" in r.message for r in caplog.records)


class TestConvertLegacyOfficeLibreOfficePresent:
    def test_reads_output_txt(self, tmp_path):
        """When LibreOffice is available, reads the converted .txt file."""
        from parsers import _platform

        dummy = str(tmp_path / "test.doc")
        Path(dummy).write_bytes(b"dummy")

        def _fake_run(cmd, **kw):
            outdir = cmd[cmd.index("--outdir") + 1]
            (Path(outdir) / "test.txt").write_text("converted text", encoding="utf-8")
            return MagicMock(returncode=0)

        with patch.object(_platform, "SYSTEM", "Linux"), \
             patch("parsers._platform._find_libreoffice", return_value="/usr/bin/soffice"), \
             patch("parsers._platform.subprocess.run", side_effect=_fake_run):
            result = _platform.convert_legacy_office(dummy, "DOC")

        assert result == "converted text"

    def test_glob_finds_txt_regardless_of_stem(self, tmp_path):
        """Output file is found by glob, not by exact stem assumption."""
        from parsers import _platform

        # Filename with special chars that LibreOffice might normalise
        dummy = str(tmp_path / "my-report (2024).doc")
        Path(dummy).write_bytes(b"dummy")

        def _fake_run(cmd, **kw):
            outdir = cmd[cmd.index("--outdir") + 1]
            # LibreOffice sanitises the stem
            (Path(outdir) / "my-report_2024_.txt").write_text("text", encoding="utf-8")
            return MagicMock(returncode=0)

        with patch.object(_platform, "SYSTEM", "Linux"), \
             patch("parsers._platform._find_libreoffice", return_value="/usr/bin/soffice"), \
             patch("parsers._platform.subprocess.run", side_effect=_fake_run):
            result = _platform.convert_legacy_office(dummy, "DOC")

        assert result == "text"


# ── reveal_in_file_manager ────────────────────────────────────────────────────

class TestRevealValidation:
    def test_rejects_relative_path(self):
        from parsers._platform import reveal_in_file_manager
        with patch("parsers._platform.subprocess.run") as mock_run:
            reveal_in_file_manager("relative/path")
        mock_run.assert_not_called()

    def test_rejects_url_scheme(self):
        from parsers._platform import reveal_in_file_manager
        with patch("parsers._platform.subprocess.run") as mock_run:
            reveal_in_file_manager("smb://server/share")
        mock_run.assert_not_called()

    def test_rejects_non_string(self):
        from parsers._platform import reveal_in_file_manager
        with patch("parsers._platform.subprocess.run") as mock_run:
            reveal_in_file_manager(None)   # type: ignore
        mock_run.assert_not_called()


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
class TestRevealDarwin:
    def test_calls_open_r(self, tmp_path):
        from parsers._platform import reveal_in_file_manager
        p = str(tmp_path / "file.md")
        with patch("parsers._platform.subprocess.run") as mock_run:
            reveal_in_file_manager(p)
        mock_run.assert_called_once_with(["/usr/bin/open", "-R", p], check=False)


class TestRevealWindows:
    def test_uses_systemroot_explorer(self, tmp_path):
        """Windows reveal passes a string command directly to CreateProcess
        (shell=False + str avoids list2cmdline double-quoting)."""
        import os
        from parsers import _platform

        p = str(tmp_path / "file.md")

        with patch.object(_platform, "SYSTEM", "Windows"), \
             patch("parsers._platform.subprocess.run") as mock_run, \
             patch.dict(os.environ, {"SystemRoot": r"C:\Windows"}):
            _platform.reveal_in_file_manager(p)

        # subprocess.run receives a single string, not a list
        cmd = mock_run.call_args[0][0]
        assert isinstance(cmd, str)
        assert os.path.join(r"C:\Windows", "explorer.exe") in cmd
        assert f'/select,"{p}"' in cmd

    def test_path_with_spaces_quoted(self, tmp_path):
        """Path with spaces is wrapped in quotes inside /select, argument."""
        import os
        from parsers import _platform

        p = str(tmp_path / "my folder" / "file.md")

        with patch.object(_platform, "SYSTEM", "Windows"), \
             patch("parsers._platform.subprocess.run") as mock_run, \
             patch.dict(os.environ, {"SystemRoot": r"C:\Windows"}):
            _platform.reveal_in_file_manager(p)

        cmd = mock_run.call_args[0][0]
        # Command is a string; path with spaces must be inside double-quotes
        assert isinstance(cmd, str)
        assert '/select,"' in cmd
        assert cmd.endswith('"')
