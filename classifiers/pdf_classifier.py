"""
PDF classifier — determines the optimal extraction strategy for a PDF.

Three-path triage:
1. Native text PDF  → extract text directly (fast, accurate)
2. Layout-broken PDF → vision + text fallback (tables crushed by text layer)
3. Scanned PDF       → vision + text fallback (image-only, no text layer)

This classifier saves unnecessary AI vision calls for native PDFs,
while ensuring scanned and layout-broken documents get proper treatment.
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

try:
    import fitz
except ImportError:
    fitz = None


class PdfType:
    NATIVE = "native"
    LAYOUT_BROKEN = "layout_broken"
    SCANNED = "scanned"


def _has_pipe_table(text):
    """Check if text contains at least one Markdown pipe table row."""
    return bool(re.search(r"\|.+\|.+\|", text))


def classify(filepath, odl_text=None):
    """
    Classify a PDF into one of three types: native, layout_broken, or scanned.

    If odl_text is provided (from opendataloader-pdf), use it for classification
    instead of PyMuPDF extraction. ODL text with pipe tables upgrades LAYOUT_BROKEN
    to NATIVE since the tables were successfully extracted.

    Returns: (pdf_type, text, metadata)
        - pdf_type: PdfType.NATIVE | LAYOUT_BROKEN | SCANNED
        - text: extracted text (may be empty for scanned)
        - metadata: dict with density, garbage_ratio, short_line_ratio, page_count
    """
    # If ODL provided good text, use it for classification
    if odl_text and len(odl_text) >= 50:
        # Get page count via fitz if available, else estimate
        pages = 1
        if fitz:
            try:
                with fitz.open(filepath) as doc:
                    pages = max(doc.page_count, 1)
            except Exception:
                pass

        density = len(odl_text.strip()) / pages
        metadata = {
            "page_count": pages,
            "char_density": round(density, 1),
            "garbage_ratio": 0,
            "short_line_ratio": 0,
            "source": "odl",
        }

        fname = os.path.basename(filepath)
        if _has_pipe_table(odl_text):
            logger.info(f"[Classifier] Native PDF via ODL (tables detected): {fname}")
            return PdfType.NATIVE, odl_text, metadata

        logger.info(f"[Classifier] Native PDF via ODL (density={density:.0f}): {fname}")
        return PdfType.NATIVE, odl_text, metadata

    # ODL not available or produced insufficient text — fall back to PyMuPDF
    if not fitz:
        logger.warning("PyMuPDF (fitz) not installed — defaulting to scanned")
        return PdfType.SCANNED, "", {"page_count": 1}

    try:
        with fitz.open(filepath) as doc:
            pages = max(doc.page_count, 1)
            parts = []
            for page in doc:
                parts.append(page.get_text())
        text = "".join(parts)
    except Exception as e:
        logger.error(f"PDF classification failed: {e}")
        return PdfType.SCANNED, "", {"page_count": 1}

    total_chars = len(text.strip())
    density = total_chars / pages
    garbage_ratio = sum(1 for c in text if _is_garbage(c)) / max(len(text), 1)

    # Layout-broken detection: >70% short lines means tables were crushed
    lines = [line for line in text.split("\n") if line.strip()]
    short_lines = sum(1 for line in lines if len(line.strip()) <= 10)
    short_line_ratio = short_lines / max(len(lines), 1)

    metadata = {
        "page_count": pages,
        "char_density": round(density, 1),
        "garbage_ratio": round(garbage_ratio, 4),
        "short_line_ratio": round(short_line_ratio, 3),
        "source": "fitz",
    }

    # Decision tree
    fname = os.path.basename(filepath)
    if density < 8:
        logger.info(f"[Classifier] Scanned PDF (density={density:.0f}): {fname}")
        return PdfType.SCANNED, text, metadata

    if garbage_ratio >= 0.05:
        logger.info(f"[Classifier] Scanned PDF (garbage={garbage_ratio:.2%}): {fname}")
        return PdfType.SCANNED, text, metadata

    if short_line_ratio > 0.70:
        logger.info(f"[Classifier] Layout-broken PDF (short_lines={short_line_ratio:.0%}): {fname}")
        return PdfType.LAYOUT_BROKEN, text, metadata

    logger.info(f"[Classifier] Native PDF (density={density:.0f}): {fname}")
    return PdfType.NATIVE, text, metadata


def _is_garbage(c):
    """Check if a character is a garbage/surrogate codepoint."""
    cp = ord(c)
    return (0xD800 <= cp <= 0xDFFF) or (0xFFF0 <= cp <= 0xFFFF)
