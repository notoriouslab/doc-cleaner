"""
DOCX parser — python-docx with table preservation, textutil (macOS) fallback.

Key feature: converts Word tables to Markdown pipe tables instead of dropping them.
"""
import os
import logging
import platform

logger = logging.getLogger(__name__)


def _table_to_markdown(table):
    """Convert a python-docx table to a Markdown pipe table.

    Heuristic for header detection: if the first row has distinct content
    from most other rows (i.e. not all-numeric), treat it as a header.
    Otherwise, generate a blank header row so the Markdown table is still valid.
    """
    if not table.rows:
        return ""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
    num_cols = len(table.rows[0].cells)
    header_sep = "| " + " | ".join(["---"] * num_cols) + " |"

    # Detect if first row looks like a header: default to True (safer for
    # CJK tables where dates like "2024-01-15" or amounts like "1,234.56"
    # would be misclassified as pure numbers by the old strip-and-isdigit check).
    # Only treat as non-header if ALL cells are empty or plain integers.
    first_cells = [cell.text.strip() for cell in table.rows[0].cells]
    all_empty_or_integer = all(
        not c or c.lstrip("-").isdigit() for c in first_cells
    )
    has_header = not all_empty_or_integer

    if has_header:
        rows.insert(1, header_sep)
    else:
        # Prepend a blank header so Markdown renderers still show a valid table
        blank_header = "| " + " | ".join([""] * num_cols) + " |"
        rows.insert(0, blank_header)
        rows.insert(1, header_sep)

    return "\n".join(rows)


def parse(filepath):
    """
    Parse DOCX file, preserving tables as Markdown pipe tables.

    Strategy:
    1. python-docx (primary): preserves table structure
    2. textutil (macOS fallback): fast but loses tables
    3. Error message if neither available
    """
    # Primary: python-docx
    try:
        from docx import Document
        from docx.text.paragraph import Paragraph
        from docx.table import Table

        doc = Document(filepath)
        parts = []
        for element in doc.element.body:
            tag = element.tag.split("}")[-1]
            if tag == "p":
                p = Paragraph(element, doc)
                if p.text.strip():
                    parts.append(p.text)
            elif tag == "tbl":
                t = Table(element, doc)
                parts.append(_table_to_markdown(t))
        if parts:
            return "\n\n".join(parts)
    except ImportError:
        logger.warning("python-docx not installed, trying textutil fallback")
    except Exception as e:
        logger.warning(f"python-docx failed: {e}, trying textutil fallback")

    # Fallback: platform-specific legacy converter
    from parsers._platform import convert_legacy_office
    return convert_legacy_office(filepath, format_label="DOCX")


def parse_doc(filepath):
    """
    Parse legacy .doc file via platform-specific converter.

    macOS: /usr/bin/textutil (system built-in)
    Windows/Linux: LibreOffice headless (if installed)
    Returns extracted plain text, or empty string on failure.
    """
    from parsers._platform import convert_legacy_office
    return convert_legacy_office(filepath, format_label="DOC")
