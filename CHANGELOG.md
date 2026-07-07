# Changelog

## v1.6.1

### Bug Fixes

- **PDF tables — merged cells**: text spanning multiple rows/columns now appears once (first cell of the span, remaining cells blank) instead of being repeated into every covered cell
- **PDF tables — cross-page data loss**: a table continuing across pages with repeated headers now merges into one continuous table with a single header; previously every continuation page was silently dropped (a 60-row table lost 23 rows)
- **PDF tables — robustness**: `|` and `\` inside cells are escaped so they can no longer corrupt the table structure; degenerate (zero-content) table detections no longer swallow the text under them or knock the whole page down to plain text
- **All table formats**: DOCX / PPTX / XLSX / Numbers now share one cell-escaping implementation — pipes, backslashes, and line breaks inside cells can no longer break table columns or split a row across lines (XLSX multi-line cells previously misaligned the whole sheet)
- **DOCX**: line breaks inside table cells are preserved as spaces (adjacent lines were concatenated without a boundary)
- **In-app preview**: table cells containing escaped pipes render as one cell instead of splitting

### Notes

- No new dependencies; no packaging changes. ODL-equipped environments (Java) are unaffected — these fixes target the built-in extraction path that packaged apps use by default.

## v1.6.0

### New Features

- **Custom output folder**: choose any folder for the converted `.md` files, in addition to "same folder" and "Desktop" (macOS GUI)
- **Remembered locations**: the chosen output folder and the last input directory persist across launches; the file picker reopens where you last were
- **Recursive folder drop**: drop a folder to convert every supported file inside it, subfolders included (1000-file cap with a notice)
- **In-app Markdown preview**: preview a converted `.md` rendered in-app (headings, pipe tables, lists, code) without leaving the app

### Security

- Preview renderer is the trust boundary: all source text is HTML-escaped, only a whitelisted tag set is emitted, links are limited to `http`/`https`/`mailto` and the `href` value is escaped — untrusted `.md` cannot inject markup into the webview
- Preview renderer guards against quadratic-backtracking input (no UI freeze on pathological Markdown) and strips NUL bytes
- Preferences loader is type-safe and never raises — a corrupt or malformed `settings.json` can never prevent the app from launching
- Recursive folder scan preserves the symlink-escape guard and does not follow symlinked subdirectories

### Improvements

- Skipped files (no extractable content) now show ⚠️ with an actionable hint, distinct from ❌ errors
- The DMG ReadMe is sourced from the single maintained `ReadMe.txt`; the packaging script reads the version from `pyproject.toml` (no hardcoded version)
- GitHub link uses a single source of truth in the bridge (no hardcoded URL in the front-end)

## v1.5.0

### New Features

- **EPUB support**: extract e-book chapters in reading (spine) order with title/author, via a self-parsed ZIP + OPF + XHTML pipeline — MIT-clean (deliberately avoids the AGPL EbookLib)
- **Apple iWork support**: `.numbers` (one section per table), `.pages`, and `.key` (Keynote — one section per slide, in reading order)
- Now **16 supported formats**

### Bug Fixes

- About panel showed version "unknown" in the packaged app — now reads the bundled `pyproject.toml`
- DMG `ReadMe.txt` packaging; clearer graceful-degradation hint for modern `.pages` files

### Docs

- Bilingual plain-text `ReadMe.txt` shipped in the DMG; positioning reframed to everyday document extraction (crediting underlying libraries)

## v1.4.0

### New Features

- **JSONL support**: convert Claude Code session transcripts (`.jsonl`) to Markdown — per-session sections with timestamps, collapsible tool-result and thinking blocks; AI structuring and PII redaction intentionally bypassed (the transcript is already structured)

### CI

- Windows MSI release upload uses the dynamic tag `${{ github.ref_name }}` instead of a hardcoded version, and ensures the release exists before upload

## v1.3.0

### New Features

- **Desktop GUI app**: macOS (universal Intel + Apple Silicon, ad-hoc signed) and Windows (MSI built via GitHub Actions) — drag-and-drop, no Python required
- Shared in-process conversion API (`core.py`); the CLI delegates to it
- Cross-platform layer (`parsers/_platform.py`): `textutil` / LibreOffice / Finder / Explorer
- PDF tables preserved as Markdown pipe tables (PyMuPDF `find_tables`); optional high-quality ODL extraction when a system Python provides it

### Security

- SSRF base-URL validation rebuilt on the `ipaddress` module (decimal/hex/partial-dotted/IPv4-mapped IPv6/ULA coverage)
- GUI symlink-escape guard, concurrency lock, and reveal-in-Finder path validation

### Bug Fixes

- DOCX merged-cell duplication (gridSpan) — use `row._tr.tc_lst`
- PDF `Col1/Col2` placeholder headers replaced with the first real data row; cross-page table de-duplication; `<br>` cleanup

## v1.2.0

### New Features

- **DXF support**: Extract text annotations, dimensions, layer names, block attributes from `.dxf` engineering drawings via `ezdxf`
- **PPTX support**: Extract slide text, tables (as Markdown pipe tables), and speaker notes from `.pptx` via `python-pptx`
- **PPT support**: Legacy `.ppt` extraction via macOS `textutil`
- **DOC support**: Legacy `.doc` extraction via macOS `textutil`

### Breaking Changes

- **YAML frontmatter**: `source_path` renamed to `sourcePath` (camelCase, consistent with `pubDate`)

### Security

- Fix YAML newline injection in `_escape_yaml_str` — `\n` and `\r` now properly escaped
- Add entity count limit (`MAX_ENTITIES=50000`) to DXF parser to prevent resource exhaustion
- Add ZIP decompressed size check (`500MB`) and slide limit (`500`) to PPTX parser
- Add `timeout=60s` to all `textutil` subprocess calls

### Improvements

- Extract shared `textutil` conversion logic into `parsers/_textutil.py` (deduplicates 3 copy-paste instances)
