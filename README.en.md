<div align="center">

# doc-cleaner

[![GitHub release](https://img.shields.io/github/v/release/notoriouslab/doc-cleaner)](https://github.com/notoriouslab/doc-cleaner/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![Supported Formats](https://img.shields.io/badge/Formats-16-orange.svg)](#supported-formats-reference)
[![Last Commit](https://img.shields.io/github/last-commit/notoriouslab/doc-cleaner)](https://github.com/notoriouslab/doc-cleaner)

**Everyday document-to-Markdown conversion — 16 formats spanning PDF, Office, Apple Keynote/Numbers, and EPUB e-books. CJK-friendly, table preservation, privacy-first, fully local.**

Part of the [notoriouslab](https://github.com/notoriouslab) open-source toolkit · Requires Python 3.9+

[中文 README](README.md)

### Download Desktop App (no Python required)

| Platform | Download | Architecture |
|----------|----------|--------------|
| **macOS** | [Doc Cleaner-1.5.0.dmg](https://github.com/notoriouslab/doc-cleaner/releases/download/v1.5.0/Doc.Cleaner-1.5.0.dmg) | Universal (Intel + Apple Silicon) |
| **Windows** | [Doc Cleaner-1.5.0.msi](https://github.com/notoriouslab/doc-cleaner/releases/download/v1.5.0/Doc.Cleaner-1.5.0.msi) | x86_64 (runs on ARM Windows too) |

> **First launch on macOS:** Right-click → Open (Ventura and earlier) or System Settings → Privacy & Security → Open Anyway (Sonoma/Sequoia)
>
> Or run once in Terminal: `xattr -cr /Applications/Doc\ Cleaner.app`

</div>

---

## Core Positioning

doc-cleaner focuses on **everyday document extraction** — turning the files you meet daily into clean, readable Markdown: PDF, Word, Excel, PowerPoint, **Apple Keynote/Numbers**, and **EPUB e-books**, with CJK text intact, tables preserved, and everything processed locally.

Few free tools combine **CJK-friendly + table preservation** with coverage of **Apple Keynote/Numbers and EPUB** — that combination is doc-cleaner's strength.

**Typical use cases:**
- 🖥️ **Desktop App** — Drag-and-drop files, zero config, double-click on macOS/Windows (non-technical users)
- 📊 **Financial statements** — Big5/CP950 auto-detected, transactions and numbers extracted intact
- 🎬 **Slides / e-books** — Keynote slides and EPUB chapters extracted to Markdown in reading order
- 📄 **Batch multi-format** — Mix 16 formats as input, unified Markdown output (CLI)
- 🔒 **Privacy-first** — `--ai none` plain text or local Ollama; documents never leave your machine
- 🤖 **AI agent integration** — OpenClaw and similar frameworks can shell-call it with `SKILL.md` support

### Three Core Strengths

| Feature | Implementation |
|---------|-----------------|
| **Most complete everyday formats** | PDF, Office, Apple Keynote/Numbers, EPUB, DXF… 16 formats, one tool |
| **CJK + tables intact** | Big5/CP950/UTF-16 auto-detect; DOCX/XLSX/PDF tables → Markdown pipe tables, numbers preserved |
| **Privacy & no-AI mode** | `--ai none` for zero API keys and zero cloud; or local Ollama inference |

---

## Quick Start

### Option A: Desktop App (non-technical users)

Download the DMG (macOS) or MSI (Windows) from the table above — no Python required.

### Option B: CLI (3 steps)

```bash
# 1. Clone
git clone https://github.com/notoriouslab/doc-cleaner.git && cd doc-cleaner

# 2. Install
pip install -r requirements.txt

# 3. Run
python cleaner.py --input ./documents/ --ai none
```

**Output:** `.md` files in `./output/` for each input file

### Common Usage Paths

**Path 1: Plain text extraction (no API key required)**
```bash
# Simplest option, zero cost
python cleaner.py --input statement.pdf --ai none
```

**Path 2: Gemini for higher quality (cloud recommended)**
```bash
cp .env.example .env
# Edit .env, add your GEMINI_API_KEY
python cleaner.py --input statement.pdf --ai gemini
```

**Path 3: Local Ollama (privacy-first)**
```bash
# Requires Ollama installed and running (see Ollama recommendations below)
python cleaner.py --input statement.pdf --ai ollama
```

**Path 4: Preview before processing**
```bash
python cleaner.py --input ./documents/ --dry-run --verbose
```

### Optional Advanced Install

High-quality PDF table extraction, decryption, PPTX/DXF support, etc.:

```bash
# High-quality PDF extraction (recommended)
pip install opendataloader-pdf            # Requires Java 11+

# PDF vision mode (scanned PDFs)
pip install pdf2image                     # Also requires: brew install poppler

# PDF decryption
pip install pikepdf

# Extra formats (PPTX / DXF)
pip install python-pptx ezdxf
```

Set API keys (if using cloud backend):
```bash
cp config.example.json config.json
cp .env.example .env
# Edit .env with GEMINI_API_KEY or GROQ_API_KEY
```

---

## Core Concepts

### Smart PDF Triage

Not all PDFs are equal. doc-cleaner auto-classifies before processing:

| Type | Characteristics | Strategy |
|------|-----------------|----------|
| **Native text** | char density ≥8, garbage <5%, short lines ≤70% | Direct extraction (fast, free) |
| **Layout-broken** | >70% short lines (tables crushed) | opendataloader-pdf table extraction / AI vision + text |
| **Scanned images** | char density <8 | PDF-to-image + AI vision |

**Cost-effective workflow:**
```bash
# Step 1: Extract all in no-AI mode (fast, free, private)
python cleaner.py --input ./documents/ --ai none --output-dir ./output/raw

# Step 2: Re-process only scanned files with AI
python cleaner.py --input scanned.pdf --ai gemini
```

### Ad Cleaning

Taiwan bank statement PDFs often have investment notices, legal disclaimers, or promotions. Two mechanisms:

| Mechanism | Behavior | Scenario |
|-----------|----------|----------|
| **Tail truncation** | Remove everything after first match | End-of-document disclaimers |
| **Inline removal** | Remove individual matched paragraphs | Promotional blocks in the middle |

In `config.json`:
```json
{
  "ad_truncation_patterns": ["謹慎理財.{0,20}信用至上"],
  "ad_strip_patterns": ["※運動賺回饋"]
}
```

Safety: if truncation would remove >70% of content, it's skipped with a warning. All regex validated at startup.

### Table Preservation

Tables are first-class citizens:

- **DOCX**: `python-docx` extracts directly → Markdown pipe tables
- **XLSX/CSV**: `pandas.to_markdown()` — all sheets preserved
- **PDF**: opendataloader-pdf produces proper pipe tables (no AI needed)
- **AI prompt**: explicitly instructs to keep existing tables unchanged

### Privacy and Security

| Option | Effect |
|--------|--------|
| `--ai none` | Zero API keys, zero cloud — local extraction only |
| `--ai ollama` | Local Ollama inference, documents stay on your machine |
| `--ai gemini` / `--ai groq` | Cloud inference, higher quality |

Other safeguards:
- **Atomic writes** — temp file + `os.replace()`, no partial output
- **Secret isolation** — API keys in `.env` only, startup validation
- **OOM protection** — PDF vision defaults to 15 pages max
- **JSON fallback** — if AI returns invalid JSON, degrades to raw text

---

## Advanced Reference

### CLI Options

```
python cleaner.py [options]

  --input, -i       File or directory to process (required, non-recursive)
  --output-dir, -o  Output directory (default: ./output)
  --config          Config file path (default: <script-dir>/config.json)
  --ai              gemini | groq | ollama | none (default: config or gemini)
  --password        PDF decryption password (overrides .env and config)
  --summary         Output JSON summary to stdout (for scripts/agents)
  --dry-run         Preview without writing
  --verbose         Debug logging
  --version         Version info
```

**Exit codes:** `0` = success · `1` = partial failure · `2` = config error

### Configuration (config.json)

```jsonc
{
  "ai": {
    "backend": "gemini",                      // default backend
    "prompt_template": "prompts/default.txt", // prompt path
    "gemini": { "model": "gemini-2.5-pro" },
    "groq": {
      "model": "meta-llama/llama-4-scout-17b-16e-instruct",
      "timeout": 120
    },
    "ollama": {
      "model": "qwen3.5:9b",
      "host": "http://localhost:11434"
    }
  },
  "pdf": {
    "dpi": 200,
    "max_pages": 15
  },
  "output": { "frontmatter": true },
  "ad_truncation_patterns": ["謹慎理財.{0,20}信用至上"],
  "ad_strip_patterns": ["※運動賺回饋"]
}
```

**Secret management:** API keys belong in `.env`, **never** in `config.json`. Validated at startup.

```bash
# .env example
GEMINI_API_KEY=...
GROQ_API_KEY=...
PDF_PASSWORD=...
```

### Custom AI Prompt Templates

doc-cleaner includes 2 templates:

| File | Purpose |
|------|---------|
| `prompts/default.txt` | General document cleaning |
| `prompts/finance.txt` | Bank statements, financial reports |

**Create your own:** in `prompts/`, AI must output JSON:

```json
{
  "title": "Short title",
  "summary": "1-2 sentence summary",
  "refined_markdown": "Full cleaned Markdown",
  "tags": ["tag1", "tag2"]
}
```

### Ollama Model Recommendations

Table reconstruction is demanding; small models struggle. If your machine has resources, qwen3.5 series natively supports vision:

| Model | Size | Vision | Tables | CJK | Notes |
|-------|------|--------|--------|-----|-------|
| `qwen3.5:27b` | 17 GB | ✓ | Good | Excellent | Best results |
| `qwen3.5:9b` | 6.6 GB | ✓ | Fair | Good | **Default**, balanced |
| `qwen3.5:4b` | 3.4 GB | ✓ | Poor | Fair | Lightweight |
| `qwen3:30b` | 19 GB | — | Good | Excellent | MoE, fast, no vision |

**Recommendation:** `qwen3.5:9b` handles scanned PDFs; `qwen3:30b` for fast inference on native-text-only. 8GB RAM users should use `--ai gemini` or `--ai none`.

### Supported Formats Reference

| Format | Parser | Tables | Notes |
|--------|--------|--------|-------|
| **PDF (native)** | PyMuPDF find_tables() / opendataloader-pdf | pipe tables | find_tables needs no extra install; ODL needs Java |
| **PDF (scanned)** | pdf2image → AI vision | AI rebuild | Needs poppler (optional) |
| **PDF (encrypted)** | pikepdf | pipe tables | Optional |
| **DOCX** | python-docx | pipe tables | Cross-platform |
| **XLSX / XLS** | pandas + xlrd | pipe tables | All sheets |
| **CSV** | pandas | pipe tables | Auto-encoding detection |
| **PPTX** | python-pptx | pipe tables | Slides + speaker notes |
| **PPT** | macOS textutil / LibreOffice | — | macOS built-in; Windows needs LibreOffice |
| **DOC** | macOS textutil / LibreOffice | — | macOS built-in; Windows needs LibreOffice |
| **DXF** | ezdxf | — | Engineering: annotations, dimensions |
| **TXT / MD** | stdlib | — | Big5/CP950/UTF-16 |
| **JSONL** | built-in | — | Claude Code session transcript → Markdown |
| **NUMBERS** | numbers-parser | pipe tables | Apple spreadsheet, one section per table |
| **KEY** | keynote-parser | — | Apple Keynote, one section per slide (IWA) |
| **PAGES** | QuickLook PDF | — | Apple Pages; modern files need Export → PDF first |
| **EPUB** | built-in (lxml) | — | E-book, one section per chapter, with title/author |

---

## Integration & Ecosystem

### AI Agent Frameworks

doc-cleaner is a standard CLI — any AI agent framework can shell-call it. Ships with `SKILL.md` for [OpenClaw](https://openclaw.ai/).

```bash
# Agent example: process + JSON summary
python cleaner.py --input document.pdf --ai none --summary
```

`--summary` output:
```json
{"version":"1.0.0","total":1,"success":1,"failed":0,"files":[{"file":"document.pdf","output":"./output/document.md","status":"ok"}]}
```

### notoriouslab Pipeline

```
gmail-statement-fetcher  →  Auto-fetch PDFs from Gmail
          ↓
    doc-cleaner          →  PDF/DOCX/XLSX → structured Markdown
          ↓
   personal-cfo          →  Monthly audit + retirement planning (in development)
```

Each tool stands alone; together they form a complete personal finance pipeline.

---

## Contributing

The easiest contributions:

1. **Add ad regex patterns** for your bank — add rules to `config.example.json`
2. **Add prompt templates** — create a `.txt` file in `prompts/`
3. **Report encoding issues** — include anonymized samples and logs

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=notoriouslab/doc-cleaner&type=Date)](https://star-history.com/#notoriouslab/doc-cleaner&Date)

---

## License

This project's code is licensed under **MIT**.

### Bundled open-source libraries

doc-cleaner stands on these libraries; listed honestly with their licenses:

| Library | Role | License |
|---------|------|---------|
| PyMuPDF | PDF parsing & table detection | **AGPL-3.0 / commercial dual license** |
| python-docx | DOCX | MIT |
| pandas · openpyxl · xlrd | XLSX / XLS / CSV | BSD / MIT |
| python-pptx | PPTX | MIT |
| ezdxf | DXF | MIT |
| numbers-parser | Apple Numbers | MIT |
| keynote-parser | Apple Keynote | MIT |
| lxml | EPUB / XML parsing | BSD |
| Pillow | image handling | HPND |
| pywebview | desktop GUI | BSD |
| tabulate | Markdown table output | MIT |

> PyMuPDF is dual-licensed under AGPL-3.0 / commercial; this project's source is public on GitHub, satisfying AGPL's source-availability requirement for distribution.
