"""
DOCX parser — python-docx with table preservation, textutil (macOS) fallback.

Key feature: converts Word tables to Markdown pipe tables instead of dropping them.
"""
import os
import logging
import platform

logger = logging.getLogger(__name__)


def _table_to_markdown(table):
    """Convert a python-docx table to a Markdown pipe table."""
    rows = []
    for row in table.rows:
        cells = [cell.text.strip().replace("|", "\\|") for cell in row.cells]
        rows.append("| " + " | ".join(cells) + " |")
    if len(rows) >= 1:
        header_sep = "| " + " | ".join(["---"] * len(table.rows[0].cells)) + " |"
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

    # Fallback: textutil (macOS only, loses table structure)
    if platform.system() == "Darwin":
        try:
            import subprocess
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_txt = os.path.join(tmpdir, "out.txt")
                subprocess.run(
                    ["textutil", "-convert", "txt", filepath, "-output", temp_txt],
                    check=True,
                    capture_output=True,
                )
                if os.path.exists(temp_txt):
                    with open(temp_txt, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        return content
        except Exception as e:
            logger.warning(f"textutil failed: {e}")

    return ""
