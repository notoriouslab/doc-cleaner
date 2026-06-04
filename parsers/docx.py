"""
DOCX parser — python-docx with table preservation, textutil (macOS) fallback.

Key feature: converts Word tables to Markdown pipe tables instead of dropping them.
"""
import os
import logging
import platform

logger = logging.getLogger(__name__)


_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _tc_text(tc):
    """Extract plain text from a <w:tc> XML element, joining paragraphs with a space."""
    parts = []
    for p in tc.iter(f"{{{_W}}}p"):
        para = "".join(t.text or "" for t in p.iter(f"{{{_W}}}t")).strip()
        if para:
            parts.append(para)
    return " ".join(parts)


def _tc_gridspan(tc):
    """Return the horizontal span (gridSpan) of a <w:tc> element (default 1)."""
    gs = tc.find(f".//{{{_W}}}gridSpan")
    if gs is None:
        return 1
    val = gs.get(f"{{{_W}}}val")
    try:
        return int(val)
    except (TypeError, ValueError):
        return 1


def _table_to_markdown(table):
    """Convert a python-docx table to a Markdown pipe table.

    Uses row._tr.tc_lst (the actual <w:tc> elements) instead of row.cells so
    that horizontally-merged cells (gridSpan) are shown exactly once, with
    empty cells filling the remaining span, rather than being repeated for
    every column they cover.
    """
    if not table.rows:
        return ""

    rows_data = []   # list of cell-text lists
    for row in table.rows:
        cells = []
        for tc in row._tr.tc_lst:
            text = _tc_text(tc).replace("|", "\\|").replace("\n", " ")
            span = _tc_gridspan(tc)
            cells.append(text)
            for _ in range(span - 1):
                cells.append("")   # blank filler for merged span
        rows_data.append(cells)

    # Determine column count from the widest row
    num_cols = max((len(r) for r in rows_data), default=0)
    if num_cols == 0:
        return ""

    # Pad shorter rows to the same width
    for r in rows_data:
        r.extend([""] * (num_cols - len(r)))

    rows = ["| " + " | ".join(r) + " |" for r in rows_data]
    header_sep = "| " + " | ".join(["---"] * num_cols) + " |"

    # Header detection: treat as non-header only if ALL unique cells are
    # empty or plain integers (avoids mis-classifying CJK header rows).
    first_unique = [_tc_text(tc).strip() for tc in table.rows[0]._tr.tc_lst]
    all_empty_or_integer = all(not c or c.lstrip("-").isdigit() for c in first_unique)
    has_header = not all_empty_or_integer

    if has_header:
        rows.insert(1, header_sep)
    else:
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
