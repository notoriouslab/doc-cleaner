"""
doc-cleaner macOS GUI — pywebview desktop app.

Entry point: main()
Bridge:      Api class (exposed as pywebview.api in JS)
"""
import json
import locale
import os
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import webview

import cleaner
import core as _core
from macapp import mdpreview, settings

GITHUB_URL = "https://github.com/notoriouslab/doc-cleaner"

SUPPORTED_TYPES = (
    "支援格式 (*.pdf;*.docx;*.xlsx;*.xls;*.csv;*.txt;*.md;*.pptx;*.dxf;*.doc;*.ppt;*.jsonl;*.numbers;*.pages;*.key;*.epub)",
    "All files (*.*)",
)

_URL_WHITELIST = {GITHUB_URL}

# Static, safe HTML shown in the preview overlay when a file can't be rendered.
_PREVIEW_ERROR_HTML = '<p class="meta">⚠️ 無法預覽此檔案 / Cannot preview this file</p>'


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
    """Detect system locale; return 'zh' for zh-* or Chinese locales, else 'en'."""
    try:
        loc, _ = locale.getlocale()
        if loc:
            loc_lower = loc.lower()
            if loc_lower.startswith("zh") or "chinese" in loc_lower:
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
    /* Markdown preview overlay */
    #preview-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,.45); z-index: 110;
      align-items: center; justify-content: center;
    }
    #preview-overlay.visible { display: flex; }
    .preview-panel {
      background: white; border-radius: 14px; padding: 0;
      width: 88%; max-width: 760px; max-height: 82vh;
      display: flex; flex-direction: column;
      box-shadow: 0 8px 32px rgba(0,0,0,.22);
    }
    .preview-head {
      display: flex; justify-content: flex-end; padding: 10px 12px;
      border-bottom: 1px solid #f2f2f7; flex-shrink: 0;
    }
    .preview-body { padding: 18px 24px; overflow: auto; }
    .preview-body .meta { color: #636366; font-size: 13px; margin: 0 0 12px; }
    .preview-body h1, .preview-body h2, .preview-body h3,
    .preview-body h4, .preview-body h5, .preview-body h6 { margin: 14px 0 8px; line-height: 1.3; }
    .preview-body p, .preview-body li, .preview-body blockquote { font-size: 14px; line-height: 1.6; }
    .preview-body blockquote { border-left: 3px solid #d1d1d6; margin: 8px 0; padding: 2px 12px; color: #636366; }
    .preview-body code { background: #f2f2f7; border-radius: 4px; padding: 1px 5px; font-size: 13px; }
    .preview-body pre { background: #f2f2f7; border-radius: 8px; padding: 12px; overflow: auto; }
    .preview-body pre code { background: none; padding: 0; }
    .preview-body table { border-collapse: collapse; margin: 10px 0; font-size: 13px; }
    .preview-body th, .preview-body td { border: 1px solid #d1d1d6; padding: 5px 10px; text-align: left; }
    .preview-body th { background: #f2f2f7; }
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
    <div id="notice" style="display:none;margin-top:6px;font-size:12px;color:#b8860b"></div>
  </div>

  <div class="card">
    <div class="output-mode">
      <span data-i18n="outputLabel">輸出位置：</span>
      <label><input type="radio" name="mode" value="sibling" checked> <span data-i18n="outputSibling">同資料夾</span></label>
      <label><input type="radio" name="mode" value="desktop"> <span data-i18n="outputDesktop">桌面</span></label>
      <label><input type="radio" name="mode" value="custom"> <span data-i18n="outputCustom">選擇資料夾…</span></label>
    </div>
    <div id="custom-dir-row" style="display:none;margin-top:6px;font-size:12px;color:#636366">
      <span id="custom-dir-path" style="word-break:break-all"></span>
      <button type="button" id="btn-change-folder" data-i18n="changeFolder" style="margin-left:6px;background:none;border:none;color:#007aff;cursor:pointer;font-size:12px;padding:0;text-decoration:underline">變更…</button>
    </div>
  </div>

  <div class="card">
    <div class="output-mode">
      <span data-i18n="formatLabel">輸出格式：</span>
      <label><input type="radio" name="format" value="md" checked> <span data-i18n="formatMd">Markdown (.md)</span></label>
      <label><input type="radio" name="format" value="epub"> <span data-i18n="formatEpub">EPUB 電子書 (.epub)</span></label>
      <label><input type="radio" name="format" value="both"> <span data-i18n="formatBoth">雙格式 (.md + .epub)</span></label>
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
      <div class="about-desc" data-i18n="aboutDesc" style="color:#636366;font-size:13px;margin:-8px 0 16px;line-height:1.5">日常文件轉 Markdown · 中文友好 · 表格保留 · 16 種格式</div>
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

  <!-- Markdown preview overlay -->
  <div id="preview-overlay">
    <div class="preview-panel">
      <div class="preview-head">
        <button class="btn-secondary" id="btn-preview-close" data-i18n="aboutClose" style="padding:6px 18px">關閉</button>
      </div>
      <div class="preview-body" id="preview-body"></div>
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
        outputCustom: '選擇資料夾…',
        changeFolder: '變更…',
        formatLabel:  '輸出格式：',
        formatMd:     'Markdown (.md)',
        formatEpub:   'EPUB 電子書 (.epub)',
        formatBoth:   '雙格式 (.md + .epub)',
        convert:      '轉換',
        revealInFinder:'在 Finder 顯示',
        preview:      '預覽',
        preparing:    '準備中…',
        done:         '完成',
        aboutDesc:    '日常文件轉 Markdown · 中文友好 · 表格保留 · 16 種格式',
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
        outputCustom: 'Choose folder…',
        changeFolder: 'Change…',
        formatLabel:  'Format:',
        formatMd:     'Markdown (.md)',
        formatEpub:   'EPUB E-book (.epub)',
        formatBoth:   'Both (.md + .epub)',
        convert:      'Convert',
        revealInFinder:'Show in Finder',
        preview:      'Preview',
        preparing:    'Preparing…',
        done:         'Done',
        aboutDesc:    'Everyday documents → Markdown · CJK-friendly · tables preserved · 16 formats',
        aboutVersion: 'Version',
        aboutLicense: 'License',
        aboutAuthor:  'Author',
        aboutClose:   'Close'
      }
    };

    var _lang = 'zh';
    var selectedPaths = [];
    var customDir = null;   // remembered custom output folder (D2)

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
      // Restore saved preferences (D1/D2)
      pywebview.api.get_prefs().then(function(prefs) {
        if (!prefs) return;
        customDir = prefs.custom_output_dir || null;
        var mode = prefs.output_mode || 'sibling';
        var radio = document.querySelector('input[name=mode][value=' + mode + ']');
        if (radio) radio.checked = true;
        showCustomDir();
        document.getElementById('custom-dir-row').style.display =
          (mode === 'custom') ? '' : 'none';

        var fmt = prefs.output_format || 'md';
        var fmtRadio = document.querySelector('input[name=format][value=' + fmt + ']');
        if (fmtRadio) fmtRadio.checked = true;
      });
    }

    // ── custom output folder (D2) ───────────────────────────────────────────
    function showCustomDir() {
      var el = document.getElementById('custom-dir-path');
      if (el) el.textContent = customDir || '';
    }

    function chooseOutputFolder() {
      pywebview.api.pick_output_folder().then(function(dir) {
        if (dir) {
          customDir = dir;
          showCustomDir();
          document.getElementById('custom-dir-row').style.display = '';
        } else if (!customDir) {
          // Cancelled with no prior folder → revert to sibling
          document.querySelector('input[name=mode][value=sibling]').checked = true;
          document.getElementById('custom-dir-row').style.display = 'none';
          pywebview.api.set_output_mode('sibling');
        }
      });
    }

    // notice area (cap notice + stale-folder fallback), localized by kind
    function onNotice(kind, n) {
      var msg = '';
      if (kind === 'cap') {
        msg = (_lang === 'zh') ? ('只載入前 ' + n + ' 個檔案') : ('Loaded first ' + n + ' files');
      } else if (kind === 'fallbackSibling') {
        msg = (_lang === 'zh')
          ? '輸出資料夾不存在，已改存到原始檔案所在資料夾'
          : 'Output folder missing; saved beside source files instead';
      }
      var el = document.getElementById('notice');
      el.textContent = msg;
      el.style.display = msg ? '' : 'none';
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

    // ── output mode change (D2: persist + custom folder picker) ─────────────
    document.querySelectorAll('input[name=mode]').forEach(function(r) {
      r.addEventListener('change', function() {
        var mode = this.value;
        document.getElementById('custom-dir-row').style.display =
          (mode === 'custom') ? '' : 'none';
        if (mode === 'custom' && !customDir) {
          chooseOutputFolder();
        } else {
          pywebview.api.set_output_mode(mode);
        }
      });
    });

    document.querySelectorAll('input[name=format]').forEach(function(r) {
      r.addEventListener('change', function() {
        pywebview.api.set_output_format(this.value);
      });
    });

    // "變更…" — re-open the folder picker to change an already-chosen folder.
    document.getElementById('btn-change-folder').addEventListener('click', function() {
      chooseOutputFolder();
    });

    // ── convert ────────────────────────────────────────────────────────────
    document.getElementById('btn-convert').addEventListener('click', function() {
      var mode = document.querySelector('input[name=mode]:checked').value;
      var format = document.querySelector('input[name=format]:checked').value;
      document.getElementById('results').innerHTML = '';
      document.getElementById('progress-label').textContent = STRINGS[_lang].preparing;
      document.getElementById('btn-convert').disabled = true;
      document.getElementById('btn-pick').disabled = true;
      pywebview.api.convert(selectedPaths, mode, (mode === 'custom') ? customDir : null, format);
    });

    // ── GitHub header button (D5) ──────────────────────────────────────────
    document.getElementById('btn-github').addEventListener('click', function() {
      pywebview.api.open_github();
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
      pywebview.api.open_github();
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
      // Three-way: skipped (no extractable content + actionable hint) is not a
      // failure — show ⚠️, distinct from a real ❌ error.
      var icon = ok ? '✅' : (result.status === 'skipped' ? '⚠️' : '❌');
      li.innerHTML =
        '<span class="icon">' + icon + '</span>' +
        '<div class="file-info">' +
          '<div class="file-name">' + escHtml(result.file) + '</div>' +
          (result.error ? '<div class="file-err">' + escHtml(result.error) + '</div>' : '') +
        '</div>';
      if (ok && result.output) {
        var pv = document.createElement('button');
        pv.className = 'btn-reveal';
        pv.textContent = STRINGS[_lang].preview;
        pv.dataset.path = result.output;
        pv.addEventListener('click', function() {
          openPreview(this.dataset.path);
        });
        li.appendChild(pv);

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

    // ── Markdown preview overlay ───────────────────────────────────────────
    function openPreview(path) {
      pywebview.api.preview_markdown(path).then(function(html) {
        // Trust boundary: `html` is the Python renderer's escaped, whitelisted
        // output (macapp/mdpreview.py). Intentionally NOT run through escHtml —
        // re-escaping would show the tags as text. Do not "fix" by wrapping.
        document.getElementById('preview-body').innerHTML = html;
        document.getElementById('preview-body').scrollTop = 0;
        document.getElementById('preview-overlay').classList.add('visible');
      });
    }
    document.getElementById('btn-preview-close').addEventListener('click', function() {
      document.getElementById('preview-overlay').classList.remove('visible');
    });
    document.getElementById('preview-overlay').addEventListener('click', function(e) {
      if (e.target === this) this.classList.remove('visible');  // scrim click closes
    });

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
        self._settings = settings.load()  # D1: never raises; defaults on missing/corrupt

    def get_app_info(self):
        """Return app metadata dict for the About overlay (D3)."""
        return self._app_info

    # ── preferences (D1/D2) ──────────────────────────────────────────────
    def get_prefs(self):
        """Return the persisted preferences for the front-end to restore on launch."""
        return {
            "output_mode": self._settings.get("output_mode", "sibling"),
            "custom_output_dir": self._settings.get("custom_output_dir"),
            "last_input_dir": self._settings.get("last_input_dir"),
            "output_format": self._settings.get("output_format", "md"),
        }

    def set_output_mode(self, mode):
        """Persist the chosen output mode (sibling/desktop/custom)."""
        self._settings["output_mode"] = mode
        settings.save(self._settings)

    def set_output_format(self, format_val):
        """Persist the chosen output format (md/epub/both)."""
        self._settings["output_format"] = format_val
        settings.save(self._settings)

    def pick_output_folder(self):
        """Open a native folder dialog (seeded at the last custom folder).

        Returns the chosen absolute path (or "" if cancelled). On success,
        persists output_mode="custom" and the chosen folder."""
        seed = self._settings.get("custom_output_dir") or ""
        result = self._window.create_file_dialog(
            webview.FileDialog.FOLDER,
            directory=seed,
        )
        chosen = (list(result)[0] if result else "")
        if chosen:
            self._settings["output_mode"] = "custom"
            self._settings["custom_output_dir"] = chosen
            settings.save(self._settings)
        return chosen

    def _remember_input_dir(self, path):
        """Persist the directory of a picked/dropped source (D2). A dropped
        folder is itself the source directory; a file's directory is its parent."""
        try:
            directory = str(Path(path) if os.path.isdir(path) else Path(path).parent)
        except (TypeError, ValueError):
            return
        if directory and directory != self._settings.get("last_input_dir"):
            self._settings["last_input_dir"] = directory
            settings.save(self._settings)

    def open_url(self, url):
        """Open URL in system browser (D5). Silently rejects unlisted URLs."""
        if url not in _URL_WHITELIST:
            return
        if sys.platform == "darwin":
            subprocess.run(["/usr/bin/open", url], check=False)
        elif sys.platform == "win32":
            os.startfile(url)
        else:
            subprocess.run(["xdg-open", url], check=False)

    def open_github(self):
        """Open the project's GitHub page. Single source of truth for the URL
        (the front-end no longer hardcodes it), so it can't drift from the
        open_url allowlist."""
        self.open_url(GITHUB_URL)

    def pick_files(self):
        """Open native file dialog (seeded at the last input dir, D2).
        Returns list of absolute path strings; remembers the source folder."""
        result = self._window.create_file_dialog(
            webview.FileDialog.OPEN,
            directory=self._settings.get("last_input_dir") or "",
            allow_multiple=True,
            file_types=SUPPORTED_TYPES,
        )
        paths = list(result) if result else []
        if paths:
            self._remember_input_dir(paths[0])
        return paths

    def convert(self, paths, output_mode, custom_dir=None, output_format="md"):
        """Start batch conversion on a daemon background thread (non-blocking).
        If a batch is already running the call is silently ignored."""
        if not self._batch_lock.acquire(blocking=False):
            return
        threading.Thread(
            target=self._run_batch_guarded,
            args=(paths, output_mode, custom_dir, output_format),
            daemon=True,
        ).start()

    def _run_batch_guarded(self, paths, output_mode, custom_dir=None, output_format="md"):
        try:
            self._run_batch(paths, output_mode, custom_dir, output_format)
        finally:
            self._batch_lock.release()

    def _run_batch(self, paths, output_mode, custom_dir=None, output_format="md"):
        """
        Build config+backend once, loop per file pushing progress to JS.
        Resolves symlinks before processing (mirrors CLI's security policy).
        """
        total = len(paths)
        config, ai_backend, prompt = _core._build_env(ai="none")
        desktop = str(Path.home() / "Desktop")

        # D2: custom output dir; if it's gone, fall back to sibling + notify once.
        if output_mode == "custom" and not (custom_dir and os.path.isdir(custom_dir)):
            output_mode = "sibling"
            if self._window:
                self._window.evaluate_js("onNotice('fallbackSibling')")

        def _resolver(path):
            if output_mode == "desktop":
                return desktop
            if output_mode == "custom" and custom_dir:  # custom_dir validated above
                return custom_dir
            return str(Path(path).parent)  # sibling, and defensive fallback

        for i, path in enumerate(paths):
            path = str(Path(path).resolve())
            self._window.evaluate_js(f"onProgress({i + 1}, {total})")
            result = _core._run_one(path, ai_backend, prompt, config, _resolver(path), output_format=output_format)
            self._window.evaluate_js(
                f"onResult({json.dumps(result, ensure_ascii=False)})"
            )

        self._window.evaluate_js("onComplete()")

    def get_dropped_paths(self):
        """Return supported file paths from a drop event (D3).

        Directories are expanded recursively into their supported files; loose
        files are kept if supported. Remembers the source folder and notifies
        the front-end when a folder expansion hits the recursion cap."""
        from webview.dom import _dnd_state
        raw = [p for _name, p in _dnd_state["paths"]]
        _dnd_state["paths"].clear()

        expanded = []
        capped = False
        for p in raw:
            if os.path.isdir(p):
                # Use the recursive collector's accurate capped flag (not a
                # len()>=cap heuristic, which false-positives at exactly the cap).
                collected, was_capped = cleaner._collect_dir_recursive(p, os.path.realpath(p))
                if was_capped:
                    capped = True
                expanded.extend(collected)
            elif os.path.isfile(p):
                if os.path.splitext(p)[1].lower() in cleaner.SUPPORTED_EXTENSIONS:
                    expanded.append(os.path.realpath(p))

        # De-duplicate, preserving order.
        seen = set()
        result = []
        for p in expanded:
            if p not in seen:
                seen.add(p)
                result.append(p)

        if raw:
            self._remember_input_dir(raw[0])
        if capped and self._window:
            self._window.evaluate_js(f"onNotice('cap', {cleaner.MAX_RECURSIVE_FILES})")
        return result

    def reveal_in_finder(self, path):
        """Reveal file in platform file manager."""
        from parsers._platform import reveal_in_file_manager
        reveal_in_file_manager(path)

    def preview_markdown(self, path):
        """Read a produced .md and return safe rendered HTML for the preview
        overlay (D5). Path-checked (absolute, existing regular file) and
        size-capped before reading; never raises — returns an escaped error
        message string on any failure."""
        try:
            if not path or not os.path.isabs(path):
                return _PREVIEW_ERROR_HTML
            real = os.path.realpath(path)
            if not os.path.isfile(real):
                return _PREVIEW_ERROR_HTML
            if os.path.getsize(real) > mdpreview.MAX_PREVIEW_BYTES:
                return _PREVIEW_ERROR_HTML
            with open(real, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            return mdpreview.render(text)
        except Exception:
            return _PREVIEW_ERROR_HTML


def main():
    from webview.dom import _dnd_state
    _dnd_state["num_listeners"] = 1

    # Detect locale and build app info once (D1, D3)
    lang = _detect_lang() if sys.platform in ("darwin", "win32") else "en"
    is_macos = sys.platform in ("darwin", "win32")
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
