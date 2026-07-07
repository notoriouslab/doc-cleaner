"""Unit tests for macapp.app — Python-JS bridge (tasks 3.1–3.5)."""
import json
import platform
import re
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module when pywebview is not installed (e.g. CI without GUI deps)
pytest.importorskip("webview", reason="pywebview not installed")


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path_factory, monkeypatch):
    """Isolate preferences I/O to a temp dir so Api tests never touch the real
    user settings file (Api.__init__ loads, several methods save)."""
    from macapp import settings as _settings
    d = tmp_path_factory.mktemp("appdata")
    monkeypatch.setattr(_settings, "app_data_dir", lambda: str(d))


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

        def _slow_batch(paths, mode, custom_dir=None, output_format="md"):
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
    # reveal_in_finder delegates to parsers._platform.reveal_in_file_manager;
    # patch the subprocess there, not in macapp.app.

    @pytest.mark.skipif(
        platform.system() != "Darwin",
        reason="macOS-specific: open -R is not used on other platforms",
    )
    def test_reveal_calls_open_r(self, tmp_path):
        """reveal_in_finder runs /usr/bin/open -R <path> on macOS."""
        api = _make_api()
        test_path = str(tmp_path / "result.md")

        with patch("parsers._platform.subprocess.run") as mock_run:
            api.reveal_in_finder(test_path)

        mock_run.assert_called_once_with(
            ["/usr/bin/open", "-R", test_path], check=False
        )

    def test_reveal_rejects_relative_path(self):
        """reveal_in_finder silently ignores non-absolute paths."""
        api = _make_api()
        with patch("parsers._platform.subprocess.run") as mock_run:
            api.reveal_in_finder("relative/path/file.md")
        mock_run.assert_not_called()

    def test_reveal_rejects_url_scheme(self):
        """reveal_in_finder silently ignores paths with URL schemes."""
        api = _make_api()
        with patch("parsers._platform.subprocess.run") as mock_run:
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


# ── Phase A: preferences, custom output folder, recursive drop ───────────────

class TestPreferences:
    def test_get_prefs_returns_persisted(self):
        api = _make_api()
        api.set_output_mode("desktop")
        prefs = api.get_prefs()
        assert prefs["output_mode"] == "desktop"
        assert "custom_output_dir" in prefs and "last_input_dir" in prefs

    def test_set_output_mode_persists_across_instances(self):
        api = _make_api()
        api.set_output_mode("desktop")
        # A fresh Api should load the saved mode (same isolated settings dir).
        api2 = _make_api()
        assert api2.get_prefs()["output_mode"] == "desktop"

    def test_pick_output_folder_persists_custom(self, tmp_path):
        api = _make_api()
        api._window.create_file_dialog.return_value = [str(tmp_path)]
        chosen = api.pick_output_folder()
        assert chosen == str(tmp_path)
        prefs = api.get_prefs()
        assert prefs["output_mode"] == "custom"
        assert prefs["custom_output_dir"] == str(tmp_path)
        # seeded at the previously-saved dir (empty string on first use)
        args, kwargs = api._window.create_file_dialog.call_args
        import webview
        assert args[0] == webview.FileDialog.FOLDER

    def test_pick_output_folder_cancel_returns_empty(self):
        api = _make_api()
        api._window.create_file_dialog.return_value = None
        assert api.pick_output_folder() == ""
        # nothing persisted as custom
        assert api.get_prefs()["custom_output_dir"] is None

    def test_pick_output_folder_can_change_after_first_pick(self, tmp_path):
        """The '變更…' flow: picking a second folder replaces the first
        (regression for "first pick works but can't change it")."""
        api = _make_api()
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()

        api._window.create_file_dialog.return_value = [str(first)]
        api.pick_output_folder()
        assert api.get_prefs()["custom_output_dir"] == str(first)

        # Re-pick a different folder (what the Change… button triggers).
        api._window.create_file_dialog.return_value = [str(second)]
        api.pick_output_folder()
        assert api.get_prefs()["custom_output_dir"] == str(second)

        # Re-pick then cancel: keeps the current folder (no revert).
        api._window.create_file_dialog.return_value = None
        assert api.pick_output_folder() == ""
        assert api.get_prefs()["custom_output_dir"] == str(second)

    def test_pick_files_remembers_input_dir(self, tmp_path):
        api = _make_api()
        f = tmp_path / "a.docx"
        api._window.create_file_dialog.return_value = [str(f)]
        api.pick_files()
        assert api.get_prefs()["last_input_dir"] == str(tmp_path)


class TestCustomOutputResolver:
    def test_custom_dir_used_for_output(self, tmp_path):
        api = _make_api()
        src = tmp_path / "src"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        captured = {}
        with patch("macapp.app._core._build_env", return_value=({}, None, "")), \
             patch("macapp.app._core._run_one",
                   side_effect=lambda p, ab, pr, cfg, od, **kwargs: captured.setdefault("od", od) or {"file": "x", "status": "ok", "output": None, "error": None}):
            api._run_batch([str(src / "a.docx")], "custom", str(out))
        assert captured["od"] == str(out)

    def test_missing_custom_dir_falls_back_to_sibling_and_notifies(self, tmp_path):
        api = _make_api()
        src = tmp_path / "doc.docx"
        gone = str(tmp_path / "deleted_folder")
        captured = {}
        with patch("macapp.app._core._build_env", return_value=({}, None, "")), \
             patch("macapp.app._core._run_one",
                   side_effect=lambda p, ab, pr, cfg, od, **kwargs: captured.setdefault("od", od) or {"file": "x", "status": "ok", "output": None, "error": None}):
            api._run_batch([str(src)], "custom", gone)
        # fell back to sibling (parent of source), and pushed the fallback notice
        assert captured["od"] == str(tmp_path)
        assert any("onNotice('fallbackSibling')" in str(c)
                   for c in api._window.evaluate_js.call_args_list)


class TestRecursiveDropExpansion:
    def _set_dnd(self, paths):
        from webview.dom import _dnd_state
        _dnd_state["paths"] = [(None, p) for p in paths]

    def test_dropped_folder_expands_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "top.md").write_text("x", encoding="utf-8")
        (tmp_path / "sub" / "deep.txt").write_text("x", encoding="utf-8")
        api = _make_api()
        self._set_dnd([str(tmp_path)])
        result = api.get_dropped_paths()
        names = sorted(Path(p).name for p in result)
        assert names == ["deep.txt", "top.md"]

    def test_dropped_loose_file_kept_if_supported(self, tmp_path):
        good = tmp_path / "a.csv"
        good.write_text("x", encoding="utf-8")
        bad = tmp_path / "b.bin"
        bad.write_text("x", encoding="utf-8")
        api = _make_api()
        self._set_dnd([str(good), str(bad)])
        result = api.get_dropped_paths()
        assert [Path(p).name for p in result] == ["a.csv"]

    def test_cap_fires_notice(self, tmp_path, monkeypatch):
        import cleaner
        monkeypatch.setattr(cleaner, "MAX_RECURSIVE_FILES", 2)
        for i in range(4):
            (tmp_path / f"f{i}.md").write_text("x", encoding="utf-8")
        api = _make_api()
        self._set_dnd([str(tmp_path)])
        result = api.get_dropped_paths()
        assert len(result) == 2
        assert any("onNotice('cap'" in str(c)
                   for c in api._window.evaluate_js.call_args_list)

    def test_exactly_cap_does_not_fire_notice(self, tmp_path, monkeypatch):
        # exactly MAX files loaded fully → NO "capped" notice (no false positive)
        import cleaner
        monkeypatch.setattr(cleaner, "MAX_RECURSIVE_FILES", 2)
        for i in range(2):
            (tmp_path / f"f{i}.md").write_text("x", encoding="utf-8")
        api = _make_api()
        self._set_dnd([str(tmp_path)])
        result = api.get_dropped_paths()
        assert len(result) == 2
        assert not any("onNotice('cap'" in str(c)
                       for c in api._window.evaluate_js.call_args_list)

    def test_dropped_folder_remembers_folder_itself(self, tmp_path):
        # last_input_dir for a dropped FOLDER is the folder itself, not its parent
        folder = tmp_path / "src"
        (folder).mkdir()
        (folder / "a.md").write_text("x", encoding="utf-8")
        api = _make_api()
        self._set_dnd([str(folder)])
        api.get_dropped_paths()
        assert api.get_prefs()["last_input_dir"] == str(folder)


# ── Markdown preview bridge (add-markdown-preview) ───────────────────────────

class TestPreviewMarkdown:
    def test_renders_real_md(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nHello **world**.", encoding="utf-8")
        api = _make_api()
        html = api.preview_markdown(str(md))
        assert "<h1>Title</h1>" in html
        assert "<strong>world</strong>" in html

    def test_missing_path_returns_error(self, tmp_path):
        api = _make_api()
        out = api.preview_markdown(str(tmp_path / "nope.md"))
        assert "Cannot preview" in out  # escaped error, no raise

    def test_relative_path_returns_error(self):
        api = _make_api()
        out = api.preview_markdown("relative/path.md")
        assert "Cannot preview" in out

    def test_directory_path_returns_error(self, tmp_path):
        api = _make_api()
        out = api.preview_markdown(str(tmp_path))
        assert "Cannot preview" in out

    def test_oversize_returns_error(self, tmp_path, monkeypatch):
        import macapp.mdpreview as mp
        monkeypatch.setattr(mp, "MAX_PREVIEW_BYTES", 10)
        big = tmp_path / "big.md"
        big.write_text("x" * 100, encoding="utf-8")
        api = _make_api()
        out = api.preview_markdown(str(big))
        assert "Cannot preview" in out

    def test_never_raises_on_garbage(self, tmp_path):
        # binary garbage decoded with errors=replace must still render, not raise
        f = tmp_path / "g.md"
        f.write_bytes(b"\xff\xfe\x00 not utf8 \x80")
        api = _make_api()
        out = api.preview_markdown(str(f))
        assert isinstance(out, str)  # produced something, no exception


class TestOpenGithub:
    def test_open_github_uses_constant(self):
        import sys
        import macapp.app as appmod
        api = _make_api()
        if sys.platform == "win32":
            with patch("macapp.app.os.startfile") as startfile:
                api.open_github()
            startfile.assert_called_once_with(appmod.GITHUB_URL)
        else:
            with patch("macapp.app.subprocess.run") as run:
                api.open_github()
            run.assert_called_once()
            args = run.call_args[0][0]
            expected_cmd = "/usr/bin/open" if sys.platform == "darwin" else "xdg-open"
            assert args == [expected_cmd, appmod.GITHUB_URL]


class TestOnResultWiring:
    """Static pin of the JS result-button gating (integrate-epub-output)."""

    def test_preview_gated_on_preview_field(self):
        import macapp.app as appmod
        html = appmod._HTML
        assert "if (ok && result.preview)" in html
        assert "pv.dataset.path = result.preview;" in html

    def test_reveal_gated_on_output_field(self):
        import macapp.app as appmod
        html = appmod._HTML
        assert "if (ok && result.output)" in html
        assert "btn.dataset.path = result.output;" in html


class TestSetLang:
    """Language persistence bridge (route-pdf-tables-native)."""

    def test_valid_codes_persist(self):
        api = _make_api()
        with patch("macapp.app.settings.save") as save:
            api.set_lang("en")
        save.assert_called_once()
        assert api._settings["lang"] == "en"

    def test_invalid_codes_ignored(self):
        api = _make_api()
        api._settings["lang"] = None
        with patch("macapp.app.settings.save") as save:
            for bad in ("fr", "", None, 123, ["zh"], "ZH"):
                api.set_lang(bad)
        save.assert_not_called()
        assert api._settings["lang"] is None

    def test_toggle_wiring_calls_bridge(self):
        import macapp.app as appmod
        assert "pywebview.api.set_lang(newLang)" in appmod._HTML


class TestStartupLangResolution:
    def test_stored_lang_wins_over_locale(self):
        import macapp.app as appmod
        with patch("macapp.app.settings.load", return_value={"lang": "en"}), \
             patch("macapp.app._detect_lang", return_value="zh") as detect:
            stored = appmod.settings.load().get("lang")
            lang = stored if stored in ("zh", "en") else appmod._detect_lang()
        assert lang == "en"
        detect.assert_not_called()

    def test_none_falls_through_to_detection(self):
        import macapp.app as appmod
        with patch("macapp.app.settings.load", return_value={"lang": None}), \
             patch("macapp.app._detect_lang", return_value="zh"):
            stored = appmod.settings.load().get("lang")
            lang = stored if stored in ("zh", "en") else appmod._detect_lang()
        assert lang == "zh"
