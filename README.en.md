# doc-cleaner

Convert PDF, DOCX, XLSX, and text files to clean, structured Markdown — CJK-friendly, table-friendly, privacy-first.

**Requires Python 3.9+** · Part of the [notoriouslab](https://github.com/notoriouslab) open-source toolkit.

> [中文 README](README.md)

---

## Why This Tool

Most document-to-Markdown tools either drop tables, butcher CJK text, or require cloud uploads. doc-cleaner was built for Traditional Chinese financial documents from day one, but works great with any language. It also integrates with AI agent frameworks (OpenClaw, etc.) — any agent can call it via shell, and it ships with a `SKILL.md` for direct use with [OpenClaw](https://openclaw.ai/).

| Feature | |
|---|---|
| CJK-first | Big5, CP950, UTF-16 auto-detection — covers all Taiwan bank statements |
| Table preservation | DOCX + XLSX → Markdown pipe tables |
| Smart PDF triage | Auto-classifies: native text / layout-broken / scanned |
| AI structuring | Gemini (cloud), Groq (cloud), or Ollama (local) |
| No-AI mode | `--ai none` — pure extraction, zero API keys, zero cloud |
| PDF decryption | Optional pikepdf |
| Ad truncation | Configurable regex patterns to strip bank disclaimers |
| Privacy-first | Local Ollama option — documents never leave your machine |
| Atomic writes | Temp file + `os.replace()` — no partial output |
| Dry-run preview | `--dry-run` before committing |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/notoriouslab/doc-cleaner.git
cd doc-cleaner

# 2. Install core dependencies
pip install -r requirements.txt

# 3. (Optional) Install AI backend
pip install google-genai python-dotenv   # for Gemini
# or
# Groq uses its OpenAI-compatible API directly; just set GROQ_API_KEY
# or
pip install ollama                        # for local Ollama

# 4. (Optional) Install PDF extras
pip install pikepdf                       # PDF decryption
pip install pdf2image                     # PDF vision mode (requires poppler)

# 5. Configure
cp config.example.json config.json
cp .env.example .env
# Edit .env — set GEMINI_API_KEY or GROQ_API_KEY if using a cloud backend

# 6. Run
python cleaner.py --input statement.pdf
# Output: ./output/statement.md
```

### No-AI Mode (Simplest)

No API keys, no cloud — just text and table extraction:

```bash
pip install -r requirements.txt
python cleaner.py --input ./downloads/ --ai none
```

### Dry Run

Preview which files would be processed without writing anything:

```bash
python cleaner.py --input ./downloads/ --dry-run --verbose
```

---

## CLI Options

```
python cleaner.py [options]

  --input, -i       File or directory to process (required, non-recursive)
  --output-dir, -o  Output directory (default: ./output)
  --config          Path to config JSON (default: <script-dir>/config.json)
  --ai              gemini | groq | ollama | none (default: from config or gemini)
  --password        PDF decryption password (overrides .env and config)
  --summary         Print JSON summary to stdout after processing (for scripts and AI agents)
  --dry-run         Preview without writing files
  --verbose         Enable debug logging
  --version         Print version and exit
```

### Exit Codes

| Code | Meaning |
|---|---|
| 0 | All files processed successfully |
| 1 | Some files failed (partial success) |
| 2 | No processable files found or config error |

---

## Configuration

The main config file is `config.json` (copy from `config.example.json`):

```jsonc
{
  "ai": {
    "backend": "gemini",                        // default AI backend
    "prompt_template": "prompts/default.txt",   // prompt template path
    "gemini": {
      "model": "gemini-2.5-pro"
    },
    "groq": {
      "model": "meta-llama/llama-4-scout-17b-16e-instruct",
      "base_url": "https://api.groq.com/openai/v1",
      "timeout": 120
    },
    "ollama": {
      "model": "qwen3.5:9b",
      "host": "http://localhost:11434"
    }
  },
  "pdf": {
    "dpi": 200,                                 // vision mode resolution
    "max_pages": 15                             // vision mode page cap (OOM protection)
  },
  "output": {
    "frontmatter": true                         // include YAML frontmatter in output
  },
  "ad_truncation_patterns": [                   // ad truncation regex (see below)
    "<投資人權益通知訊息[ >]",
    "謹慎理財.{0,20}信用至上"
  ]
}
```

### Secret Management

- **API keys and passwords** belong in `.env` only — **never** in `config.json`
- Both `config.json` and `.env` are excluded via `.gitignore`
- doc-cleaner warns at startup if it detects secret-like fields in `config.json`

```
# .env example
GEMINI_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
PDF_PASSWORD=your-pdf-password
```

Password priority: `--password` CLI arg > `.env` (`PDF_PASSWORD`) > `config.json`

---

## Custom Ad Truncation Patterns

Taiwan bank statement PDFs often end with investment risk notices, legal disclaimers, and promotional boilerplate. doc-cleaner truncates everything after the first regex match.

Patterns are sourced from two places:

| Location | Purpose |
|---|---|
| `classifiers/noise.py` `DEFAULT_CUTOFF_PATTERNS` | Built-in defaults for common Taiwan financial documents |
| `config.json` `ad_truncation_patterns` | **User-customizable** — overrides the built-in defaults |

**Add your bank's patterns**: edit the `ad_truncation_patterns` array in `config.json`:

```json
"ad_truncation_patterns": [
  "<投資人權益通知訊息[ >]",
  "[～~]\\s*總\\s*公\\s*司\\s*訊\\s*息\\s*[～~]",
  "依金融監督管理委員會保險局",
  "謹慎理財.{0,20}信用至上",
  "本商品由.{0,30}核准",
  "your new pattern here"
]
```

Safety: if truncation would remove more than 70% of content, it's skipped with a warning.

All regex patterns are validated at startup — invalid syntax causes an immediate error, not a mid-processing crash.

---

## Custom AI Prompt Templates

doc-cleaner ships with two prompt templates:

| File | Purpose |
|---|---|
| `prompts/default.txt` | General-purpose document cleaning |
| `prompts/finance.txt` | Bank statements and financial reports (preserves transactions, amounts) |

Switch in `config.json`:

```json
"ai": { "prompt_template": "prompts/finance.txt" }
```

**Write your own**: create a `.txt` file in `prompts/`. The AI must output JSON with these fields:

```json
{
  "title": "Short descriptive title",
  "summary": "1-2 sentence summary",
  "refined_markdown": "Full cleaned Markdown content",
  "tags": ["tag1", "tag2"]
}
```

Example: for medical documents, create `prompts/medical.txt` and emphasize preserving patient IDs, dates, and diagnosis codes.

---

## Smart PDF Triage

Not all PDFs are equal. doc-cleaner classifies each PDF before processing:

| Type | Detection | Strategy |
|---|---|---|
| Native text | char density ≥8, garbage <5%, short lines ≤70% | Direct text extraction (fast, free) |
| Layout-broken | >70% short lines (tables crushed) | AI vision + text fallback |
| Scanned | char density <8 | AI vision + text fallback |

This saves AI calls for native-text PDFs — only files that truly need vision processing get sent to the AI.

### Hybrid Strategy (Recommended)

The most cost-effective workflow:

```bash
# Step 1: Extract everything in raw mode (fast, free, private)
python cleaner.py --input ./downloads/ --ai none --output-dir ./output/raw

# Step 2: Re-process only files that need AI
# (check logs for "Layout-broken" or "Scanned" classification)
python cleaner.py --input problem_file.pdf --ai gemini --output-dir ./output/ai
# or
python cleaner.py --input problem_file.pdf --ai groq --output-dir ./output/ai
```

---

## Table Preservation

Tables are first-class citizens:

- **DOCX**: `python-docx` extracts tables → Markdown pipe tables (`|` delimiters)
- **XLSX/CSV**: `pandas.to_markdown()` — all sheets, empty cells filled, capped at 8000 chars/sheet
- **AI prompt**: explicitly instructs "keep existing pipe tables EXACTLY as-is"

---

## Ollama Model Recommendations

Table reconstruction from layout-broken PDFs is demanding. Smaller models will struggle. Tested on MacBook Air M2 (8GB) and iMac 2019 — neither performed well with local Ollama, but if your machine has more RAM, the qwen3.5 series supports vision (Image) natively — ideal for scanned PDFs:

| Model | Size | Vision | Table reconstruction | CJK quality | Notes |
|---|---|---|---|---|---|
| `qwen3.5:27b` | 17 GB | Yes | Good | Excellent | **Recommended** — native vision, 256K context |
| `qwen3.5:9b` | 6.6 GB | Yes | Fair | Good | **Default** — runs on most machines, handles scanned PDFs |
| `qwen3.5:4b` | 3.4 GB | Yes | Poor | Fair | Text OK, tables marginal |
| `qwen3:30b` | 19 GB | No | Good | Excellent | MoE, fast inference, but no vision |

> **Recommendation**: prefer the `qwen3.5` series — native vision means scanned PDFs can send images directly to the model without extra OCR. `qwen3.5:27b` gives the best results; `qwen3.5:9b` (6.6GB) is the default, balancing quality and resource requirements.
>
> If you don't need to process scanned PDFs (only native-text PDFs, DOCX, XLSX), `qwen3:30b` with MoE architecture offers faster inference.
>
> **8GB RAM users**: Ollama will be slow. Use `--ai gemini` or `--ai none` instead.

---

## Supported Formats

| Format | Parser | Tables | Notes |
|---|---|---|---|
| **PDF** (native text) | PyMuPDF (fitz) | AI rebuild | Fast text extraction; tables need AI structuring |
| **PDF** (scanned) | pdf2image → AI vision | AI rebuild | Requires poppler |
| **PDF** (encrypted) | pikepdf → above | AI rebuild | Optional pikepdf |
| **DOCX** | python-docx | pipe tables | Cross-platform; textutil fallback on macOS only |
| **XLSX / XLS** | pandas + openpyxl | pipe tables | All sheets |
| **CSV** | pandas | pipe tables | Auto-detected |
| **TXT / MD** | stdlib | — | Multi-encoding (Big5, CP950, UTF-16) |

### Installing poppler

PDF vision mode (converting scanned PDF pages to images) requires the poppler system package:

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils
```

If you don't need vision mode, use `--ai none` to skip it entirely.

---

## Security

- **No cloud required**: `--ai ollama` or `--ai none` keeps everything local
- **Atomic writes**: temp file + `os.replace()` prevents partial output
- **Secret isolation**: API keys in `.env` only (never `config.json`), startup validation
- **OOM protection**: PDF vision capped at 15 pages by default (configurable)
- **Ad truncation guard**: truncation skipped if it would remove >70% of content
- **JSON graceful degradation**: if AI returns unparseable JSON, falls back to raw text mode

See [SECURITY.md](SECURITY.md) for the full security policy.

---

## AI Agent Integration (OpenClaw, etc.)

doc-cleaner is a standard CLI tool — any AI agent framework can call it via shell. It ships with a `SKILL.md` for direct use with [OpenClaw](https://openclaw.ai/).

```bash
# Agent usage: process file + get machine-readable summary
python cleaner.py --input document.pdf --ai none --summary
```

`--summary` output example:
```json
{"version":"1.0.0","total":1,"success":1,"failed":0,"files":[{"file":"document.pdf","output":"./output/document.md","status":"ok"}]}
```

Agents can use exit codes to determine success (0=all OK, 1=partial failure, 2=config error) and parse the `--summary` JSON for per-file results.

---

## Part of the notoriouslab Pipeline

```
gmail-statement-fetcher   →  Auto-download PDF statements from Gmail
        ↓
   doc-cleaner             →  PDF/DOCX/XLSX → structured Markdown
        ↓
   personal-cfo            →  Monthly audit + retirement glide path (in development)
```

Each tool works standalone. Together they form a full personal finance automation pipeline.

---

## Contributing

The easiest contributions:

1. **Add ad truncation patterns** for your bank — add regex to `config.example.json`
2. **Add prompt templates** for your document type — create a `.txt` in `prompts/`
3. **Report encoding issues** with anonymized samples and logs

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT
