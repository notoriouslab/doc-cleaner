---
name: doc-cleaner
description: Convert PDF, DOCX, XLSX, and text files to clean, structured Markdown. CJK-friendly, table-friendly, privacy-first.
version: 1.0.0
metadata: {"openclaw":{"emoji":"📄","homepage":"https://github.com/notoriouslab/doc-cleaner","requires":{"bins":["python3"]}}}
---

# doc-cleaner

Convert documents (PDF, DOCX, XLSX, TXT) to clean, structured Markdown.

## When to use

- User asks to convert a document to Markdown
- User wants to extract text or tables from PDF/DOCX/XLSX files
- User wants to clean up bank statements or financial documents
- User asks to process a batch of documents in a directory

## Commands

### Convert a single file (no AI, fastest)
```bash
python3 {baseDir}/cleaner.py --input "{{file_path}}" --ai none
```

### Convert a single file with AI structuring
```bash
python3 {baseDir}/cleaner.py --input "{{file_path}}" --ai gemini
```

### Convert a single file with Groq structuring
```bash
python3 {baseDir}/cleaner.py --input "{{file_path}}" --ai groq
```

### Convert all files in a directory
```bash
python3 {baseDir}/cleaner.py --input "{{directory}}" --ai none --output-dir "{{output_dir}}"
```

### Preview without writing (dry run)
```bash
python3 {baseDir}/cleaner.py --input "{{file_path}}" --dry-run --verbose
```

### Get machine-readable result summary
```bash
python3 {baseDir}/cleaner.py --input "{{file_path}}" --ai none --summary
```

The `--summary` flag prints a JSON summary to stdout after processing:
```json
{"version":"1.0.0","total":3,"success":2,"failed":1,"files":[{"file":"report.pdf","output":"./output/report.md","status":"ok"},{"file":"scan.pdf","output":null,"status":"no_content"},{"file":"data.xlsx","output":"./output/data.md","status":"ok"}]}
```

## Options

| Flag | Description |
|---|---|
| `--input, -i` | File or directory to process (required, non-recursive) |
| `--output-dir, -o` | Output directory (default: ./output) |
| `--ai` | `gemini`, `groq`, `ollama`, or `none` (default: from config or gemini) |
| `--password` | PDF decryption password |
| `--config` | Path to config JSON |
| `--summary` | Print JSON summary to stdout after processing |
| `--dry-run` | Preview without writing files |
| `--verbose` | Enable debug logging |

## Supported formats

PDF (native, scanned, encrypted), DOCX, XLSX, XLS, CSV, TXT, MD

## Exit codes

| Code | Meaning |
|---|---|
| 0 | All files processed successfully |
| 1 | Some files failed (partial success) |
| 2 | No processable files found or config error |

## Notes

- Output defaults to `./output/` relative to current directory
- For scanned PDFs, AI mode (`gemini`, `groq`, or `ollama`) gives much better results
- `--ai none` requires zero API keys and zero network access
- CJK encoding (Big5, CP950, UTF-16) is auto-detected
- Tables in DOCX and XLSX are preserved as Markdown pipe tables
