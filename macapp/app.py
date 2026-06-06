"""
doc-cleaner macOS GUI — pywebview desktop app.

Entry point: main()
Bridge:      Api class (exposed as pywebview.api in JS)
"""
import json
import locale
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import webview

import core as _core

GITHUB_URL = "https://github.com/notoriouslab/doc-cleaner"

SUPPORTED_TYPES = (
    "支援格式 (*.pdf;*.docx;*.xlsx;*.xls;*.csv;*.txt;*.md;*.pptx;*.dxf;*.doc;*.ppt;*.jsonl;*.numbers;*.pages;*.key)",
    "All files (*.*)",
)

_URL_WHITELIST = {GITHUB_URL}


def _read_version():
    """Read version from pyproject.toml; return 'unknown' on any failure."""
    try:
        toml_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return data.get("tool", {}).get("briefcase", {}).get("version", "unknown")
    except Exception:
        return "unknown"


def _detect_lang():
    """Detect system locale; return 'zh' for zh-* locales, else 'en'."""
    try:
        loc, _ = locale.getlocale()
        if loc and loc.lower().startswith("zh"):
            return "zh"
    except Exception:
        pass
    return "en"


_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Doc Cleaner</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "PingFang TC", "Helvetica Neue", sans-serif;
      background: #f5f5f7; color: #1d1d1f; padding: 24px; min-height: 100vh;
    }
    .header-row {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 20px;
    }
    h1 { font-size: 22px; font-weight: 600; }
    .header-actions { display: flex; gap: 8px; align-items: center; }
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
    .btn-header {
      background: #f2f2f7; color: #007aff; font-size: 13px;
      padding: 6px 12px; border-radius: 8px;
    }
    .btn-header:hover { background: #e5e5ea; }
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
    /* About overlay */
    #about-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,.45); z-index: 100;
      align-items: center; justify-content: center;
    }
    #about-overlay.visible { display: flex; }
    .about-panel {
      background: white; border-radius: 16px; padding: 28px 32px;
      min-width: 300px; box-shadow: 0 8px 32px rgba(0,0,0,.22);
    }
    .about-app-name { font-size: 20px; font-weight: 700; margin-bottom: 16px; }
    .about-row {
      display: flex; justify-content: space-between; gap: 16px;
      font-size: 14px; padding: 6px 0; border-bottom: 1px solid #f2f2f7;
    }
    .about-row:last-of-type { border: none; }
    .about-row-label { color: #636366; }
    .about-row-value { color: #1d1d1f; text-align: right; }
    .btn-about-link {
      background: none; color: #007aff; font-size: 13px; padding: 0;
      border-radius: 0; text-decoration: underline; cursor: pointer;
    }
    .about-close-row { margin-top: 20px; text-align: center; }
  </style>
</head>
<body>

  <div class="header-row">
    <h1 data-i18n="title">文件清洗工具</h1>
    <div class="header-actions">
      <button class="btn-header" id="btn-github" data-i18n="github">GitHub</button>
      <button class="btn-header" id="btn-about">?</button>
      <button class="btn-header" id="btn-lang" style="display:none">EN</button>
    </div>
  </div>

  <div class="card">
    <div class="drop-zone" id="drop-zone">
      <p data-i18n="dropZoneText">📂 將文件拖放至此</p>
      <p class="hint" data-i18n="dropZoneHint">或使用下方按鈕選擇</p>
    </div>
    <div class="row">
      <button class="btn-primary" id="btn-pick" data-i18n="pickFiles">選擇檔案</button>
      <button class="btn-secondary" id="btn-clear" style="display:none" data-i18n="clearFiles">清除選擇</button>
    </div>
    <ul id="file-list"></ul>
    <div id="file-count"></div>
  </div>

  <div class="card">
    <div class="output-mode">
      <span data-i18n="outputLabel">輸出位置：</span>
      <label><input type="radio" name="mode" value="sibling" checked> <span data-i18n="outputSibling">同資料夾</span></label>
      <label><input type="radio" name="mode" value="desktop"> <span data-i18n="outputDesktop">桌面</span></label>
    </div>
  </div>

  <div class="card">
    <div id="progress-label"></div>
    <ul id="results"></ul>
  </div>

  <button class="btn-convert" id="btn-convert" disabled data-i18n="convert">轉換</button>

  <!-- About overlay (D4: in-page, no extra window) -->
  <div id="about-overlay">
    <div class="about-panel" id="about-panel">
      <div class="about-app-name">Doc Cleaner</div>
      <div class="about-row">
        <span class="about-row-label" data-i18n="aboutVersion">版本</span>
        <span class="about-row-value" id="about-ver">—</span>
      </div>
      <div class="about-row">
        <span class="about-row-label" data-i18n="aboutLicense">授權</span>
        <span class="about-row-value">MIT</span>
      </div>
      <div class="about-row">
        <span class="about-row-label" data-i18n="aboutAuthor">作者</span>
        <span class="about-row-value">notoriouslab</span>
      </div>
      <div class="about-row">
        <span class="about-row-label">GitHub</span>
        <button class="btn-about-link about-row-value" id="btn-about-github">github.com/notoriouslab/doc-cleaner</button>
      </div>
      <div class="about-close-row">
        <button class="btn-secondary" id="btn-about-close" data-i18n="aboutClose" style="padding:8px 24px">關閉</button>
      </div>
    </div>
  </div>

  <script>
    // ── i18n string table (D2) ─────────────────────────────────────────────
    var STRINGS = {
      zh: {
        title:        '文件清洗工具',
        github:       'GitHub',
        dropZoneText: '📂 將文件拖放至此',
        dropZoneHint: '或使用下方按鈕選擇',
        pickFiles:    '選擇檔案',
        clearFiles:   '清除選擇',
        outputLabel:  '輸出位置：',
        outputSibling:'同資料夾',
        outputDesktop:'桌面',
        convert:      '轉換',
        revealInFinder:'在 Finder 顯示',
        preparing:    '準備中…',
        done:         '完成',
        aboutVersion: '版本',
        aboutLicense: '授權',
        aboutAuthor:  '作者',
        aboutClose:   '關閉'
      },
      en: {
        title:        'Doc Cleaner',
        github:       'GitHub',
        dropZoneText: '📂 Drop files here',
        dropZoneHint: 'or use the button below',
        pickFiles:    'Choose Files',
        clearFiles:   'Clear',
        outputLabel:  'Output:',
        outputSibling:'Same folder',
        outputDesktop:'Desktop',
        convert:      'Convert',
        revealInFinder:'Show in Finder',
        preparing:    'Preparing…',
        done:         'Done',
        aboutVersion: 'Version',
        aboutLicense: 'License',
        aboutAuthor:  'Author',
        aboutClose:   'Close'
      }
    };

    var _lang = 'zh';
    var selectedPaths = [];

    // ── render / setLang (D2) ──────────────────────────────────────────────
    function setLang(code) {
      _lang = (STRINGS[code] ? code : 'en');
      document.querySelectorAll('[data-i18n]').forEach(function(el) {
        var key = el.getAttribute('data-i18n');
        var val = STRINGS[_lang][key];
        if (val !== undefined) el.textContent = val;
      });
      // Toggle button shows the OTHER language
      var toggle = document.getElementById('btn-lang');
      if (toggle) toggle.textContent = (_lang === 'zh' ? 'EN' : 'ZH');
    }

    // ── init — called from Python after page loads ──────────────────────────
    function init(langCode, isMacos) {
      setLang(langCode);
      if (isMacos) {
        document.getElementById('btn-lang').style.display = '';
      }
      // Pre-fetch app info for About overlay
      pywebview.api.get_app_info().then(function(info) {
        document.getElementById('about-ver').textContent = info.version;
      });
    }

    // ── file selection ─────────────────────────────────────────────────────
    function renderFiles() {
      var list  = document.getElementById('file-list');
      var count = document.getElementById('file-count');
      list.innerHTML = '';
      selectedPaths.forEach(function(p) {
        var li = document.createElement('li');
        li.title = p;
        li.textContent = p.split('/').pop();
        list.appendChild(li);
      });
      var n = selectedPaths.length;
      count.textContent = n > 0
        ? (_lang === 'zh' ? '已選 ' + n + ' 個檔案' : n + ' file(s) selected')
        : '';
      document.getElementById('btn-clear').style.display = n > 0 ? '' : 'none';
      document.getElementById('btn-convert').disabled = n === 0;
    }

    document.getElementById('btn-pick').addEventListener('click', function() {
      pywebview.api.pick_files().then(function(paths) {
        if (paths && paths.length) {
          paths.forEach(function(p) {
            if (selectedPaths.indexOf(p) === -1) selectedPaths.push(p);
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

    // ── drag-and-drop ──────────────────────────────────────────────────────
    var dz = document.getElementById('drop-zone');
    dz.addEventListener('dragover',  function(e) { e.preventDefault(); dz.classList.add('hover'); });
    dz.addEventListener('dragleave', function()  { dz.classList.remove('hover'); });
    dz.addEventListener('drop', function(e) {
      e.preventDefault();
      dz.classList.remove('hover');
      document.getElementById('file-count').textContent = _lang === 'zh' ? '讀取中…' : 'Loading…';
      pywebview.api.get_dropped_paths().then(function(paths) {
        if (paths && paths.length) {
          onDropPaths(paths);
        } else {
          document.getElementById('file-count').textContent = '';
          renderFiles();
        }
      });
    });

    // ── convert ────────────────────────────────────────────────────────────
    document.getElementById('btn-convert').addEventListener('click', function() {
      var mode = document.querySelector('input[name=mode]:checked').value;
      document.getElementById('results').innerHTML = '';
      document.getElementById('progress-label').textContent = STRINGS[_lang].preparing;
      document.getElementById('btn-convert').disabled = true;
      document.getElementById('btn-pick').disabled = true;
      pywebview.api.convert(selectedPaths, mode);
    });

    // ── GitHub header button (D5) ──────────────────────────────────────────
    document.getElementById('btn-github').addEventListener('click', function() {
      pywebview.api.open_url('https://github.com/notoriouslab/doc-cleaner');
    });

    // ── lang toggle (Manual language toggle) ──────────────────────────────
    document.getElementById('btn-lang').addEventListener('click', function() {
      setLang(_lang === 'zh' ? 'en' : 'zh');
      renderFiles(); // refresh count text
    });

    // ── About overlay (D4) ────────────────────────────────────────────────
    document.getElementById('btn-about').addEventListener('click', function() {
      document.getElementById('about-overlay').classList.add('visible');
    });
    document.getElementById('btn-about-close').addEventListener('click', function() {
      document.getElementById('about-overlay').classList.remove('visible');
    });
    document.getElementById('about-overlay').addEventListener('click', function(e) {
      // Close when clicking the scrim (not the panel itself)
      if (e.target === this) this.classList.remove('visible');
    });
    document.getElementById('btn-about-github').addEventListener('click', function() {
      pywebview.api.open_url('https://github.com/notoriouslab/doc-cleaner');
    });

    // ── Python-called callbacks ────────────────────────────────────────────
    function onProgress(current, total) {
      document.getElementById('progress-label').textContent = _lang === 'zh'
        ? '第 ' + current + '／共 ' + total + ' 個'
        : current + ' / ' + total;
    }

    function onResult(result) {
      var ul = document.getElementById('results');
      var li = document.createElement('li');
      var ok = result.status === 'ok';
      li.innerHTML =
        '<span class="icon">' + (ok ? '✅' : '❌') + '</span>' +
        '<div class="file-info">' +
          '<div class="file-name">' + escHtml(result.file) + '</div>' +
          (result.error ? '<div class="file-err">' + escHtml(result.error) + '</div>' : '') +
        '</div>';
      if (ok && result.output) {
        var btn = document.createElement('button');
        btn.className = 'btn-reveal';
        btn.textContent = STRINGS[_lang].revealInFinder;
        btn.dataset.path = result.output;
        btn.addEventListener('click', function() {
          pywebview.api.reveal_in_finder(this.dataset.path);
        });
        li.appendChild(btn);
      }
      ul.appendChild(li);
    }

    function onComplete() {
      document.getElementById('progress-label').textContent = STRINGS[_lang].done;
      document.getElementById('btn-convert').disabled = false;
      document.getElementById('btn-pick').disabled = false;
    }

    function onDropPaths(paths) {
      paths.forEach(function(p) {
        if (selectedPaths.indexOf(p) === -1) selectedPaths.push(p);
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

    def __init__(self, app_info=None):
        self._batch_lock = threading.Lock()
        self._app_info = app_info or {"version": "unknown", "author": "notoriouslab", "license": "MIT", "url": GITHUB_URL}

    def get_app_info(self):
        """Return app metadata dict for the About overlay (D3)."""
        return self._app_info

    def open_url(self, url):
        """Open URL in system browser (D5). Silently rejects unlisted URLs."""
        if url not in _URL_WHITELIST:
            return
        subprocess.run(["open", url], check=False)

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
        try:
            self._run_batch(paths, output_mode)
        finally:
            self._batch_lock.release()

    def _run_batch(self, paths, output_mode):
        """
        Build config+backend once, loop per file pushing progress to JS.
        Resolves symlinks before processing (mirrors CLI's security policy).
        """
        total = len(paths)
        config, ai_backend, prompt = _core._build_env(ai="none")
        desktop = str(Path.home() / "Desktop")

        def _resolver(path):
            return desktop if output_mode == "desktop" else str(Path(path).parent)

        for i, path in enumerate(paths):
            path = str(Path(path).resolve())
            self._window.evaluate_js(f"onProgress({i + 1}, {total})")
            result = _core._run_one(path, ai_backend, prompt, config, _resolver(path))
            self._window.evaluate_js(
                f"onResult({json.dumps(result, ensure_ascii=False)})"
            )

        self._window.evaluate_js("onComplete()")

    def get_dropped_paths(self):
        """Return file paths captured from macOS NSPasteboard after a drop event."""
        from webview.dom import _dnd_state
        paths = [p for _name, p in _dnd_state["paths"]]
        _dnd_state["paths"].clear()
        return paths

    def reveal_in_finder(self, path):
        """Reveal file in platform file manager."""
        from parsers._platform import reveal_in_file_manager
        reveal_in_file_manager(path)


def main():
    from webview.dom import _dnd_state
    _dnd_state["num_listeners"] = 1

    # Detect locale and build app info once (D1, D3)
    lang = _detect_lang() if sys.platform == "darwin" else "en"
    is_macos = sys.platform == "darwin"
    app_info = {
        "version": _read_version(),
        "author": "notoriouslab",
        "license": "MIT",
        "url": GITHUB_URL,
    }

    api = Api(app_info)
    window = webview.create_window(
        "Doc Cleaner",
        html=_HTML,
        js_api=api,
        width=600,
        height=760,
        min_size=(500, 600),
    )
    api._window = window

    # Inject initial language after page loads (D1, D2)
    is_macos_js = "true" if is_macos else "false"
    window.events.loaded += lambda: window.evaluate_js(
        f"init('{lang}', {is_macos_js})"
    )

    webview.start(debug=False)
