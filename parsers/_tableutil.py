"""
Shared table-cell normalization and escaping for Markdown pipe tables.

Single source of truth for every parser that renders pipe tables
(pdf, docx, pptx, xlsx, numbers). The rules mirror the archived
pdf-table-extraction spec: None → "", whitespace runs collapsed to single
spaces, backslash escaped before pipe so a cell ending in '\\' cannot turn
the following column delimiter into an escaped literal pipe.
"""
import re


def normalize_cell(value):
    """Normalize one cell value: None → '', str(), collapse all whitespace."""
    if value is None:
        return ""
    return re.sub(r'\s+', ' ', str(value)).strip()


def escape_cell(value):
    """Normalize + escape one cell for pipe-table rendering."""
    return normalize_cell(value).replace('\\', '\\\\').replace('|', '\\|')
