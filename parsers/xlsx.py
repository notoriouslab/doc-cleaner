"""
XLSX / CSV parser — pandas to_markdown with all sheets, openpyxl fallback.

Key features:
- All sheets rendered as separate sections
- NaN → empty string (clean tables)
- 8000 char cap per sheet (token budget)
- openpyxl fallback if pandas unavailable
"""
import os
import logging

logger = logging.getLogger(__name__)


def parse(filepath, max_chars_per_sheet=8000):
    """
    Parse Excel (.xlsx/.xls) or CSV file into Markdown pipe tables.

    All sheets are included, each as a ## section. Rows beyond the token
    budget are truncated with a note.
    """
    ext = os.path.splitext(filepath)[1].lower()

    # Primary: pandas
    try:
        import pandas as pd

        if ext == ".csv":
            # Try multiple encodings for CJK CSV files (e.g. Big5 bank statements)
            _csv_encodings = ["utf-8", "big5", "cp950", "utf-16"]
            df = None
            for enc in _csv_encodings:
                try:
                    df = pd.read_csv(filepath, encoding=enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            if df is None:
                df = pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")
            sheets = {"CSV": df}
        elif ext == ".xls":
            # xlrd 2.x only supports .xls — explicit engine prevents ImportError
            # on environments where openpyxl is present but xlrd is not
            sheets = pd.read_excel(filepath, sheet_name=None, engine="xlrd")
        else:
            sheets = pd.read_excel(filepath, sheet_name=None)

        parts = []
        for name, df in sheets.items():
            df = df.fillna("")
            md = df.to_markdown(index=False)
            if len(md) > max_chars_per_sheet:
                # Binary search for max rows that fit within the char budget
                lo, hi = 1, len(df)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if len(df.head(mid).to_markdown(index=False)) <= max_chars_per_sheet:
                        lo = mid
                    else:
                        hi = mid - 1
                md = df.head(lo).to_markdown(index=False)
                # Safety: if even 1 row exceeds budget (very wide table), hard-truncate
                if len(md) > max_chars_per_sheet:
                    md = md[:max_chars_per_sheet] + "\n\n_(content truncated due to wide table)_"
                else:
                    md += f"\n\n_(truncated: {len(df)} rows total, showing {lo} / 截斷：原始共 {len(df)} 行，顯示 {lo} 行)_"
            parts.append(f"## Sheet: {name}\n\n{md}")
        return "\n\n".join(parts)

    except ImportError:
        logger.warning("pandas not installed, falling back to openpyxl")
    except Exception as e:
        logger.warning(f"pandas parse failed: {e}, falling back to openpyxl")

    # Fallback: openpyxl (raw text, no pipe tables)
    try:
        import openpyxl

        wb = openpyxl.load_workbook(filepath, data_only=True)
        lines = []
        for sheet in wb.worksheets:
            lines.append(f"--- Sheet: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                lines.append("\t".join(str(c) if c is not None else "" for c in row))
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Excel parse failed: {e}")
        return ""
