"""
doc-cleaner macOS GUI — pywebview desktop app.

Entry point: main()
Bridge:      Api class (exposed as pywebview.api in JS)
"""
import json
import subprocess
import threading
from pathlib import Path

import webview

import core as _core

SUPPORTED_TYPES = (
    "支援格式 (*.pdf;*.docx;*.xlsx;*.xls;*.csv;*.txt;*.md;*.pptx;*.dxf;*.doc;*.ppt)",
    "All files (*.*)",
)

_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>文件清洗工具</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Helvetica Neue", sans-serif;
      background: #f5f5f7; color: #1d1d1f; padding: 24px; min-height: 100vh;
    }
    h1 { font-size: 22px; font-weight: 600; margin-bottom: 20px; }
    .card {
      background: white; border-radius: 12px; padding: 20px; margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,.1);
    }
    .drop-zone {
      border: 2px dashed #c7c7cc; border-radius: 8px; padding: 32px 20px;
      text-align: center; color: #8e8e93; transition: all .2s; cursor: pointer;
    }
    .drop-zone.hover { border-color: #007aff; background: #f0f6ff; color: #007aff; }
    .drop-zone p { margin: 8px 0; }
    .drop-zone .hint { font-size: 13px; }
    button {
      border: none; border-radius: 8px; cursor: pointer; font-size: 15px;
      padding: 10px 20px; font-family: inherit; transition: opacity .15s;
    }
    button:active { opacity: .7; }
    .btn-primary { background: #007aff; color: white; }
    .btn-secondary { background: #e5e5ea; color: #1d1d1f; }
    button:disabled { opacity: .4; cursor: default; }
    .row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
    .output-mode { display: flex; gap: 16px; align-items: center; font-size: 15px; }
    .output-mode label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
    #file-list {
      list-style: none; font-size: 13px; color: #636366; max-height: 140px;
      overflow-y: auto; margin-top: 12px;
    }
    #file-list li {
      padding: 4px 0; border-bottom: 1px solid #f2f2f7;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    #file-list li:last-child { border: none; }
    #file-count { font-size: 13px; color: #8e8e93; margin-top: 8px; }
    #progress-label { font-size: 14px; color: #636366; min-height: 20px; }
    #results { list-style: none; max-height: 320px; overflow-y: auto; }
    #results li {
      display: flex; align-items: flex-start; gap: 10px;
      padding: 10px 0; border-bottom: 1px solid #f2f2f7; font-size: 14px;
    }
    #results li:last-child { border: none; }
    .icon { font-size: 18px; line-height: 1; flex-shrink: 0; }
    .file-info { flex: 1; min-width: 0; }
    .file-name { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .file-err  { font-size: 12px; color: #ff3b30; margin-top: 2px; }
    .btn-reveal {
      background: #f2f2f7; color: #007aff; font-size: 12px;
      padding: 4px 10px; border-radius: 6px; flex-shrink: 0;
    }
    .btn-convert {
      background: #34c759; color: white; font-size: 16px;
      padding: 12px 32px; border-radius: 10px; width: 100%; margin-top: 4px;
    }
  </style>
</head>
<body>
  <h1>文件清洗工具</h1>

  <div class="card">
    <div class="drop-zone" id="drop-zone">
      <p>📂 將文件拖放至此</p>
      <p class="hint">或使用下方按鈕選擇</p>
    </div>
    <div class="row">
      <button class="btn-primary" id="btn-pick">選擇檔案</button>
      <button class="btn-secondary" id="btn-clear" style="display:none">清除選擇</button>
    </div>
    <ul id="file-list"></ul>
    <div id="file-count"></div>
  </div>

  <div class="card">
    <div class="output-mode">
      <span>輸出位置：</span>
      <label><input type="radio" name="mode" value="sibling" checked> 同資料夾</label>
      <label><input type="radio" name="mode" value="desktop"> 桌面</label>
    </div>
  </div>

  <div class="card">
    <div id="progress-label"></div>
    <ul id="results"></ul>
  </div>

  <button class="btn-convert" id="btn-convert" disabled>轉換</button>

  <script>
    var selectedPaths = [];

    function renderFiles() {
      var list = document.getElementById('file-list');
      var count = document.getElementById('file-count');
      list.innerHTML = '';
      selectedPaths.forEach(function(p) {
        var li = document.createElement('li');
        li.title = p;
        li.textContent = p.split('/').pop();
        list.appendChild(li);
      });
      var n = selectedPaths.length;
      count.textContent = n > 0 ? '已選 ' + n + ' 個檔案' : '';
      document.getElementById('btn-clear').style.display = n > 0 ? '' : 'none';
      document.getElementById('btn-convert').disabled = n === 0;
    }

    document.getElementById('btn-pick').addEventListener('click', function() {
      pywebview.api.pick_files().then(function(paths) {
        if (paths && paths.length) {
          paths.forEach(function(p) {
            if (!selectedPaths.includes(p)) selectedPaths.push(p);
          });
          renderFiles();
        }
      });
    });

    document.getElementById('btn-clear').addEventListener('click', function() {
      selectedPaths = [];
      renderFiles();
      document.getElementById('results').innerHTML = '';
      document.getElementById('progress-label').textContent = '';
    });

    // drag-and-drop (best-effort — full paths depend on platform/context)
    var dz = document.getElementById('drop-zone');
    dz.addEventListener('dragover', function(e) { e.preventDefault(); dz.classList.add('hover'); });
    dz.addEventListener('dragleave', function() { dz.classList.remove('hover'); });
    dz.addEventListener('drop', function(e) {
      e.preventDefault();
      dz.classList.remove('hover');
      document.getElementById('file-count').textContent = '讀取中…';
      // pywebview reads file URLs from macOS NSPasteboard in performDragOperation_,
      // but only when _dnd_state['num_listeners'] > 0 (set in main()).
      // We ask Python for those paths via get_dropped_paths().
      pywebview.api.get_dropped_paths().then(function(paths) {
        if (paths && paths.length) {
          onDropPaths(paths);
        } else {
          document.getElementById('file-count').textContent = '';
          renderFiles();
        }
      });
    });

    document.getElementById('btn-convert').addEventListener('click', function() {
      var mode = document.querySelector('input[name=mode]:checked').value;
      document.getElementById('results').innerHTML = '';
      document.getElementById('progress-label').textContent = '準備中…';
      document.getElementById('btn-convert').disabled = true;
      document.getElementById('btn-pick').disabled = true;
      pywebview.api.convert(selectedPaths, mode);
    });

    // called from Python via window.evaluate_js()
    function onProgress(current, total) {
      document.getElementById('progress-label').textContent =
        '第 ' + current + '／共 ' + total + ' 個';
    }

    function onResult(result) {
      var ul = document.getElementById('results');
      var li = document.createElement('li');
      var ok = result.status === 'ok';
      // Build static parts via innerHTML (no user-controlled attributes)
      li.innerHTML =
        '<span class="icon">' + (ok ? '✅' : '❌') + '</span>' +
        '<div class="file-info">' +
          '<div class="file-name">' + escHtml(result.file) + '</div>' +
          (result.error ? '<div class="file-err">' + escHtml(result.error) + '</div>' : '') +
        '</div>';
      // Reveal button uses data-path to avoid onclick attribute quoting issues
      if (ok && result.output) {
        var btn = document.createElement('button');
        btn.className = 'btn-reveal';
        btn.textContent = '在 Finder 顯示';
        btn.dataset.path = result.output;
        btn.addEventListener('click', function() {
          pywebview.api.reveal_in_finder(this.dataset.path);
        });
        li.appendChild(btn);
      }
      ul.appendChild(li);
    }

    function onComplete() {
      document.getElementById('progress-label').textContent = '完成';
      document.getElementById('btn-convert').disabled = false;
      document.getElementById('btn-pick').disabled = false;
    }

    // Called from Python after pywebview reads dropped file paths from macOS NSPasteboard
    function onDropPaths(paths) {
      paths.forEach(function(p) {
        if (!selectedPaths.includes(p)) selectedPaths.push(p);
      });
      renderFiles();
    }

    function escHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }
  </script>
</body>
</html>"""


class Api:
    """Python–JS bridge. Methods called from JS via pywebview.api.*"""

    _window = None  # assigned by main() after create_window()

    def __init__(self):
        self._batch_lock = threading.Lock()

    def pick_files(self):
        """Open native file dialog. Returns list of absolute path strings."""
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            allow_multiple=True,
            file_types=SUPPORTED_TYPES,
        )
        return list(result) if result else []

    def convert(self, paths, output_mode):
        """Start batch conversion on a daemon background thread (non-blocking).
        If a batch is already running the call is silently ignored."""
        if not self._batch_lock.acquire(blocking=False):
            return
        threading.Thread(
            target=self._run_batch_guarded,
            args=(paths, output_mode),
            daemon=True,
        ).start()

    def _run_batch_guarded(self, paths, output_mode):
        """Wrapper that releases _batch_lock when _run_batch finishes."""
        try:
            self._run_batch(paths, output_mode)
        finally:
            self._batch_lock.release()

    def _run_batch(self, paths, output_mode):
        """
        Build config+backend once, loop per file pushing progress to JS.
        Resolves symlinks before processing (mirrors CLI's security policy).
        Calls onProgress(current, total) before each file and onResult(dict) after.
        """
        total = len(paths)
        config, ai_backend, prompt = _core._build_env(ai="none")
        desktop = str(Path.home() / "Desktop")

        def _resolver(path):
            return desktop if output_mode == "desktop" else str(Path(path).parent)

        for i, path in enumerate(paths):
            path = str(Path(path).resolve())   # B2: resolve symlinks (mirrors CLI policy)
            self._window.evaluate_js(f"onProgress({i + 1}, {total})")
            result = _core._run_one(path, ai_backend, prompt, config, _resolver(path))
            self._window.evaluate_js(
                f"onResult({json.dumps(result, ensure_ascii=False)})"
            )

        self._window.evaluate_js("onComplete()")

    def get_dropped_paths(self):
        """Return file paths captured from macOS NSPasteboard after a drop event.

        pywebview's cocoa backend stores paths in _dnd_state['paths'] inside
        performDragOperation_, but only when _dnd_state['num_listeners'] > 0
        (set directly in main() at startup to avoid DOM API timing issues).
        """
        from webview.dom import _dnd_state
        paths = [p for _name, p in _dnd_state["paths"]]
        _dnd_state["paths"].clear()
        return paths

    def reveal_in_finder(self, path):
        """Reveal file in macOS Finder with open -R.
        Rejects non-absolute paths and URL schemes as a defence-in-depth measure."""
        if not isinstance(path, str) or not Path(path).is_absolute():
            return
        if "://" in path:
            return
        subprocess.run(["/usr/bin/open", "-R", path], check=False)


def main():
    # Tell pywebview's macOS backend to capture dropped file paths from NSPasteboard.
    # performDragOperation_ only stores paths when _dnd_state['num_listeners'] > 0.
    # We set this directly instead of using element.on() to avoid DOM API timing issues.
    from webview.dom import _dnd_state
    _dnd_state["num_listeners"] = 1

    api = Api()
    window = webview.create_window(
        "Doc Cleaner",
        html=_HTML,
        js_api=api,
        width=600,
        height=760,
        min_size=(500, 600),
    )
    api._window = window
    webview.start(debug=False)
