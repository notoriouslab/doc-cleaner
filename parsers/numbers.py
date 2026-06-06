"""
Apple Numbers (.numbers) parser — numbers-parser library.

Key features:
- All sheets/tables rendered as Markdown pipe tables (## heading per table)
- Per-table 8000-char truncation with note (mirrors xlsx budget)
- Graceful ImportError and file-read error handling
"""
import logging

logger = logging.getLogger(__name__)

MAX_CHARS_PER_TABLE = 8000


def _cell_str(value):
    if value is None:
        return ""
    return str(value)


def _table_to_markdown(rows, max_chars=MAX_CHARS_PER_TABLE):
    """
    Convert a list-of-lists to a Markdown pipe table.
    First row is treated as the header.
    Returns (markdown_str, total_data_rows, shown_data_rows).
    """
    if not rows:
        return "", 0, 0

    header = rows[0]
    data = rows[1:]
    total = len(data)
    n_cols = len(header)

    sep = "| " + " | ".join("---" for _ in range(n_cols)) + " |"
    header_line = "| " + " | ".join(_cell_str(c) for c in header) + " |"

    def _render(n):
        lines = [header_line, sep]
        for row in data[:n]:
            padded = list(row) + [""] * max(0, n_cols - len(row))
            lines.append("| " + " | ".join(_cell_str(c) for c in padded[:n_cols]) + " |")
        return "\n".join(lines)

    full_md = _render(total)
    if len(full_md) <= max_chars:
        return full_md, total, total

    # Binary search for max rows within budget
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(_render(mid)) <= max_chars:
            lo = mid
        else:
            hi = mid - 1

    truncated = _render(lo)
    # Safety: header+sep alone exceeds budget (pathologically wide table)
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars]
    note = f"\n\n(truncated — {lo} rows shown of {total})"
    return truncated + note, total, lo


def parse(filepath, max_chars_per_table=MAX_CHARS_PER_TABLE):
    """
    Parse Apple Numbers (.numbers) file into Markdown pipe tables.

    Each table is emitted under a ## heading:
      - Single table per sheet:   ## <SheetName>
      - Multiple tables per sheet: ## <SheetName> — <TableName>
    Empty tables are skipped. Tables exceeding max_chars_per_table are
    truncated with a note.
    """
    try:
        from numbers_parser import Document
    except ImportError:
        logger.warning(
            "numbers-parser not installed — install it with: pip install numbers-parser"
        )
        return ""

    try:
        doc = Document(filepath)
    except Exception as e:
        logger.warning(f"Failed to open Numbers file: {e}")
        return ""

    parts = []
    try:
        for sheet in doc.sheets:
            tables = list(sheet.tables)
            multi = len(tables) > 1
            for table in tables:
                try:
                    rows = [[cell.value for cell in row] for row in table.rows()]
                except Exception as e:
                    logger.warning(f"Failed to read table '{getattr(table, 'name', '?')}': {e}")
                    continue

                if not rows:
                    continue

                md, _total, _shown = _table_to_markdown(rows, max_chars=max_chars_per_table)
                if not md:
                    continue

                heading = (
                    f"## {sheet.name} — {table.name}" if multi else f"## {sheet.name}"
                )
                parts.append(f"{heading}\n\n{md}")

    except Exception as e:
        logger.warning(f"Failed to read Numbers document structure: {e}")
        return ""

    return "\n\n".join(parts)
