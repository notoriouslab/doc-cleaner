"""Unit tests for macapp.app — Python-JS bridge (tasks 3.1–3.5)."""
import json
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Skip entire module when pywebview is not installed (e.g. CI without GUI deps)
pytest.importorskip("webview", reason="pywebview not installed")


def _make_api():
    """Return an Api instance with a mocked window."""
    from macapp.app import Api
    api = Api()
    api._window = MagicMock()
    return api


# ── Task 3.1: pick_files returns real absolute paths ─────────────────────────

class TestPickFiles:
    def test_returns_paths_from_dialog(self, tmp_path):
        """pick_files returns the paths provided by create_file_dialog."""
        api = _make_api()
        expected = [str(tmp_path / "a.docx"), str(tmp_path / "b.xlsx")]
        api._window.create_file_dialog.return_value = expected

        result = api.pick_files()

        assert result == expected

    def test_uses_open_dialog_and_allow_multiple(self):
        """pick_files calls create_file_dialog with OPEN mode and allow_multiple=True."""
        import webview
        api = _make_api()
        api._window.create_file_dialog.return_value = []

        api.pick_files()

        _, kwargs = api._window.create_file_dialog.call_args
        assert kwargs.get("allow_multiple") is True
        # dialog_type should be the OPEN variant (positional arg 0 or kwarg)
        args, _ = api._window.create_file_dialog.call_args
        assert args[0] == webview.FileDialog.OPEN

    def test_returns_empty_list_when_cancelled(self):
        """pick_files returns [] when the user cancels the dialog (returns None)."""
        api = _make_api()
        api._window.create_file_dialog.return_value = None

        result = api.pick_files()

        assert result == []


# ── Task 3.2: background thread + progress pushes ────────────────────────────

class TestConvertBackgroundThread:
    def test_convert_ignored_if_batch_running(self, tmp_path):
        """Second convert() while first is running is silently ignored (no new thread)."""
        api = _make_api()
        # Pre-acquire the lock to simulate an in-progress batch
        api._batch_lock.acquire()

        batch_calls = []
        with patch.object(api, "_run_batch", side_effect=lambda *a: batch_calls.append(1)):
            api.convert([str(tmp_path / "a.docx")], "sibling")

        assert batch_calls == [], "_run_batch should NOT be called when lock is held"
        api._batch_lock.release()

    def test_convert_returns_immediately(self, tmp_path):
        """convert() starts work on a background thread and returns without blocking."""
        api = _make_api()
        started = threading.Event()
        finished = threading.Event()

        def _slow_batch(paths, mode):
            started.set()
            finished.wait(timeout=5)

        with patch.object(api, "_run_batch", side_effect=_slow_batch):
            api.convert([str(tmp_path / "x.docx")], "sibling")
            assert started.wait(timeout=2), "background thread did not start"

        finished.set()  # unblock the guarded wrapper so lock is released

    def test_run_batch_pushes_progress_before_each_file(self, tmp_path):
        """_run_batch calls onProgress(n, total) before processing each file."""
        api = _make_api()
        paths = [str(tmp_path / "a.docx"), str(tmp_path / "b.docx")]

        fake_result = {"file": "a.docx", "input": paths[0],
                       "output": None, "status": "ok", "error": None}

        with patch("macapp.app._core._build_env", return_value=({}, None, None)), \
             patch("macapp.app._core._run_one", return_value=fake_result):
            api._run_batch(paths, "sibling")

        js_calls = [c[0][0] for c in api._window.evaluate_js.call_args_list]
        progress_calls = [c for c in js_calls if c.startswith("onProgress")]
        assert "onProgress(1, 2)" in progress_calls
        assert "onProgress(2, 2)" in progress_calls

    def test_run_batch_ends_with_oncomplete(self, tmp_path):
        """_run_batch calls onComplete() after all files are processed."""
        api = _make_api()
        fake = {"file": "f.docx", "input": "/f.docx",
                "output": None, "status": "ok", "error": None}

        with patch("macapp.app._core._build_env", return_value=({}, None, None)), \
             patch("macapp.app._core._run_one", return_value=fake):
            api._run_batch(["/f.docx"], "sibling")

        last_call = api._window.evaluate_js.call_args_list[-1][0][0]
        assert last_call == "onComplete()"


# ── Task 3.3: per-file ✅/❌ result display ──────────────────────────────────

class TestResultDisplay:
    def test_onresult_called_for_each_file(self, tmp_path):
        """_run_batch calls onResult once per file with a result dict."""
        api = _make_api()
        paths = [str(tmp_path / "ok.docx"), str(tmp_path / "err.docx")]

        def _fake_run_one(path, *args, **kwargs):
            if "err" in Path(path).name:
                return {"file": Path(path).name, "input": path,
                        "output": None, "status": "error", "error": "parse failed"}
            return {"file": Path(path).name, "input": path,
                    "output": str(Path(path).with_suffix(".md")), "status": "ok", "error": None}

        with patch("macapp.app._core._build_env", return_value=({}, None, None)), \
             patch("macapp.app._core._run_one", side_effect=_fake_run_one):
            api._run_batch(paths, "sibling")

        js_calls = [c[0][0] for c in api._window.evaluate_js.call_args_list]
        result_calls = [c for c in js_calls if c.startswith("onResult")]
        assert len(result_calls) == 2

        # Parse the result dicts from the JS calls
        parsed = [json.loads(re.search(r"onResult\((.+)\)$", c).group(1))
                  for c in result_calls]
        statuses = {r["file"]: r["status"] for r in parsed}
        assert statuses["ok.docx"] == "ok"
        assert statuses["err.docx"] == "error"

    def test_error_result_includes_message(self, tmp_path):
        """Error results carry the error message in the onResult payload."""
        api = _make_api()
        err_path = str(tmp_path / "bad.pdf")

        fake_err = {"file": "bad.pdf", "input": err_path,
                    "output": None, "status": "error", "error": "corrupted file"}

        with patch("macapp.app._core._build_env", return_value=({}, None, None)), \
             patch("macapp.app._core._run_one", return_value=fake_err):
            api._run_batch([err_path], "sibling")

        js_calls = [c[0][0] for c in api._window.evaluate_js.call_args_list]
        result_call = next(c for c in js_calls if c.startswith("onResult"))
        payload = json.loads(re.search(r"onResult\((.+)\)$", result_call).group(1))
        assert payload["error"] == "corrupted file"


# ── Task 3.4: reveal_in_finder + output-mode resolver ───────────────────────

class TestRevealInFinder:
    def test_reveal_calls_open_r(self, tmp_path):
        """reveal_in_finder runs /usr/bin/open -R <path> for a valid absolute path."""
        api = _make_api()
        test_path = str(tmp_path / "result.md")

        with patch("macapp.app.subprocess.run") as mock_run:
            api.reveal_in_finder(test_path)

        mock_run.assert_called_once_with(
            ["/usr/bin/open", "-R", test_path], check=False
        )

    def test_reveal_rejects_relative_path(self):
        """reveal_in_finder silently ignores non-absolute paths."""
        api = _make_api()
        with patch("macapp.app.subprocess.run") as mock_run:
            api.reveal_in_finder("relative/path/file.md")
        mock_run.assert_not_called()

    def test_reveal_rejects_url_scheme(self):
        """reveal_in_finder silently ignores paths with URL schemes."""
        api = _make_api()
        with patch("macapp.app.subprocess.run") as mock_run:
            api.reveal_in_finder("smb://attacker.example.com/share")
        mock_run.assert_not_called()


class TestOutputModeResolver:
    def test_sibling_mode_writes_beside_source(self, tmp_path):
        """sibling mode: output_dir = parent of each source file."""
        api = _make_api()
        src = str(tmp_path / "docs" / "report.docx")

        captured_dirs = []

        def _fake_run_one(path, ai_backend, prompt, config, output_dir, **kw):
            captured_dirs.append(output_dir)
            return {"file": "report.docx", "input": path,
                    "output": None, "status": "ok", "error": None}

        with patch("macapp.app._core._build_env", return_value=({}, None, None)), \
             patch("macapp.app._core._run_one", side_effect=_fake_run_one):
            api._run_batch([src], "sibling")

        assert captured_dirs[0] == str(tmp_path / "docs")

    def test_desktop_mode_writes_to_desktop(self, tmp_path):
        """desktop mode: output_dir = ~/Desktop for every file."""
        api = _make_api()
        src = str(tmp_path / "deep" / "file.xlsx")

        captured_dirs = []

        def _fake_run_one(path, ai_backend, prompt, config, output_dir, **kw):
            captured_dirs.append(output_dir)
            return {"file": "file.xlsx", "input": path,
                    "output": None, "status": "ok", "error": None}

        with patch("macapp.app._core._build_env", return_value=({}, None, None)), \
             patch("macapp.app._core._run_one", side_effect=_fake_run_one):
            api._run_batch([src], "desktop")

        from pathlib import Path as _Path
        assert captured_dirs[0] == str(_Path.home() / "Desktop")


# ── Task 3.5: Traditional Chinese labels in HTML ─────────────────────────────

class TestTraditionalChineseUI:
    def test_required_labels_present(self):
        """All user-visible labels and buttons in _HTML are Traditional Chinese."""
        from macapp.app import _HTML

        required_tc = [
            "文件清洗工具",   # window title / h1
            "將文件拖放至此",  # drop zone
            "選擇檔案",       # pick button
            "清除選擇",       # clear button
            "輸出位置",       # output mode label
            "同資料夾",       # sibling radio
            "桌面",           # desktop radio
            "轉換",           # convert button
            "在 Finder 顯示", # reveal button
            "準備中",         # progress init
            "完成",           # onComplete
        ]
        for label in required_tc:
            assert label in _HTML, f"Missing Traditional Chinese label: {label!r}"

    def test_no_simplified_chinese_markers(self):
        """_HTML does not contain obvious Simplified Chinese substitutions."""
        from macapp.app import _HTML
        # Common Simplified replacements that differ from Traditional
        simplified = ["选择", "清除选", "转换", "桌面文件"]
        for s in simplified:
            assert s not in _HTML, f"Found Simplified Chinese: {s!r}"
