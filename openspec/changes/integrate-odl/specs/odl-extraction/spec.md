## ADDED Requirements

### Requirement: ODL text extraction

The system SHALL use opendataloader-pdf (Fast mode) as the primary PDF text extraction engine when both Java 11+ and the `opendataloader-pdf` Python package are available. The system SHALL fall back to PyMuPDF (fitz) when ODL is not available.

#### Scenario: ODL available and PDF is not encrypted

- **WHEN** a PDF file is processed AND opendataloader-pdf is installed AND Java is available
- **THEN** the system extracts text using opendataloader-pdf Fast mode AND returns Markdown-formatted text with pipe tables preserved

#### Scenario: ODL not installed

- **WHEN** a PDF file is processed AND opendataloader-pdf is NOT installed
- **THEN** the system falls back to PyMuPDF extraction AND logs a debug message indicating ODL is unavailable

#### Scenario: Java not available

- **WHEN** a PDF file is processed AND Java is NOT available on PATH
- **THEN** the system falls back to PyMuPDF extraction AND logs a debug message indicating Java is unavailable

### Requirement: ODL output post-processing

The system SHALL post-process opendataloader-pdf output to remove noise before passing it to the classifier or output stage.

#### Scenario: Image references removed

- **WHEN** ODL output contains Markdown image references matching `![image N](...)` pattern
- **THEN** the system removes all such image reference lines from the output

#### Scenario: Blank lines compressed

- **WHEN** ODL output contains three or more consecutive blank lines
- **THEN** the system compresses them to a single blank line

#### Scenario: Existing ad truncation passthrough

- **WHEN** ODL output is post-processed
- **THEN** the existing `noise.clean_text()` function SHALL be applied with the configured `ad_truncation_patterns` and `strip_urls` settings

### Requirement: Encrypted PDF handling with ODL

The system SHALL decrypt encrypted PDFs with pikepdf before passing them to ODL, using the same decryption flow as the existing PyMuPDF path.

#### Scenario: Encrypted PDF with password provided

- **WHEN** an encrypted PDF is processed AND a password is available (CLI, .env, or config)
- **THEN** the system decrypts the PDF with pikepdf first AND passes the decrypted file to ODL

#### Scenario: Encrypted PDF without password

- **WHEN** an encrypted PDF is processed AND no password is available AND ODL raises an encryption error
- **THEN** the system falls back to PyMuPDF (which handles some permission-only encryption without password)

### Requirement: Classifier integration with ODL output

The system SHALL use ODL output quality to inform the classifier decision, reducing unnecessary AI calls.

#### Scenario: ODL produces sufficient output for a previously LAYOUT_BROKEN PDF

- **WHEN** a PDF that would be classified as LAYOUT_BROKEN by PyMuPDF metrics is processed AND ODL produces structured text with at least one pipe table
- **THEN** the classifier SHALL upgrade the classification to NATIVE (no AI needed)

#### Scenario: ODL produces empty or near-empty output

- **WHEN** ODL returns fewer than 50 characters of text for a multi-page PDF
- **THEN** the classifier SHALL classify the PDF as SCANNED (AI processing needed)
