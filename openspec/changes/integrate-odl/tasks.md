## 1. ODL Text Extraction

- [x] 1.1 Add `opendataloader-pdf>=2.2.0` as optional dependency in `requirements.txt` (commented out like other optionals, with install instructions)
- [x] 1.2 Implement ODL text extraction: add `extract_text_odl(filepath)` function in `parsers/pdf.py` that calls `opendataloader_pdf.convert(filepath, format="markdown")`, reads the output .md file from disk (ODL writes to same directory as input), and returns the Markdown string. Wrap in try/except to return `None` on failure (Java missing, ODL not installed, encrypted PDF error). Log failures at debug level.
- [x] 1.3 Add `odl_available()` probe function in `parsers/pdf.py` that checks both `import opendataloader_pdf` and `java -version` subprocess call succeed. Cache result for session lifetime using a module-level variable.

## 2. ODL Output Post-processing

- [x] 2.1 Implement ODL output post-processing: add `clean_odl_output(text)` function in `parsers/pdf.py` that: (a) removes lines matching `![image \d+](...)`  regex pattern, (b) compresses 3+ consecutive blank lines to 1, (c) strips leading/trailing whitespace. Return cleaned string.
- [x] 2.2 Wire `clean_odl_output()` into `extract_text_odl()` so it runs on every ODL result before returning.

## 3. Classifier Integration with ODL Output

- [x] 3.1 Implement classifier integration with ODL output: modify `classify()` in `classifiers/pdf_classifier.py` to accept an optional `odl_text` parameter. When `odl_text` is provided and non-empty (>= 50 chars), use it for classification metrics instead of PyMuPDF extraction. When `odl_text` contains at least one pipe table (`|...|...|` pattern), upgrade LAYOUT_BROKEN classification to NATIVE.
- [x] 3.2 Modify `parse_file()` in `cleaner.py` for PDF handling: attempt ODL extraction first via `extract_text_odl()`. If ODL returns text, pass it as `odl_text` to `classify()`. If ODL returns `None`, fall through to existing PyMuPDF path unchanged. Apply `noise.clean_text()` to final text regardless of extraction source.

## 4. Encrypted PDF Handling with ODL

- [x] 4.1 Implement encrypted PDF handling with ODL: in `parse_file()` PDF branch in `cleaner.py`, move pikepdf decryption step before the ODL extraction attempt so ODL receives decrypted files. The existing decryption logic (`pdf.decrypt_pdf()`) already returns the decrypted path — pass that path to `extract_text_odl()` instead of the original.

## 5. Cleanup and Verification

- [x] 5.1 Add ODL output file cleanup in `extract_text_odl()`: after reading the .md file ODL writes to disk, delete the .md file and the `*_images/` directory (ODL creates these as side effects alongside the input PDF).
- [x] 5.2 Run doc-cleaner against the 6 test PDFs in `~/Downloads/test/` with `--ai none --verbose` and verify: (a) ODL is used for 5 non-encrypted PDFs, (b) encrypted PDF falls back to PyMuPDF path, (c) bank statements produce pipe tables in output, (d) meeting minutes produce structured headings.
