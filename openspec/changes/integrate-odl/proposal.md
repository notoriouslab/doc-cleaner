## Why

doc-cleaner currently uses PyMuPDF (fitz) for PDF text extraction. While fast, PyMuPDF produces unusable output for table-heavy documents — tables are flattened into one-value-per-line text. Benchmark testing on 6 real-world PDFs (bank statements, annual reports, church meeting minutes) showed opendataloader-pdf (ODL) produces properly structured Markdown tables (pipe tables) where PyMuPDF completely fails. This eliminates the need to send table-heavy PDFs to expensive AI vision processing.

## What Changes

- Add opendataloader-pdf as an optional PDF extraction backend in `parsers/pdf.py`
- Add a Python post-processor to clean ODL output (strip `![image ...]` references, compress blank lines)
- Adjust `pdf_classifier.py` to evaluate ODL output quality before deciding whether AI processing is needed
- Keep PyMuPDF as fallback when Java/ODL is not installed
- Add `opendataloader-pdf` as optional dependency in `requirements.txt`

## Non-Goals

- Not replacing PyMuPDF entirely — it remains as fallback for environments without Java
- Not integrating ODL Hybrid mode — Fast mode is sufficient; complex scanned documents still go to AI
- Not changing the AI backend architecture — the "human in the loop" (Claude CLI) workflow is a user-level pattern, not a code-level change
- Not adding Java installation automation — users install Java themselves

## Capabilities

### New Capabilities

- `odl-extraction`: opendataloader-pdf Fast mode extraction with Python post-processing (image reference removal, blank line compression, ad truncation passthrough)

### Modified Capabilities

(none — existing classifier and AI fallback logic are implementation details, not spec-level changes)

## Impact

- Affected code: `parsers/pdf.py`, `classifiers/pdf_classifier.py`, `cleaner.py` (minor), `requirements.txt`
- New dependency: `opendataloader-pdf>=2.2.0` (optional, requires Java 11+)
- No API/CLI interface changes — existing `--ai none` workflow benefits automatically
