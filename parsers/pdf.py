"""
PDF parser — decrypt + native text extraction + image extraction for vision mode.

Supports:
- High-quality extraction via opendataloader-pdf (optional, preferred)
- Decryption via pikepdf (optional, graceful skip)
- Native text extraction via PyMuPDF (fitz) as fallback
- Image extraction via pdf2image (optional, for AI vision mode)
- Image optimization (RGB, max 1600px) to save tokens
"""
import os
import re
import sys
import shutil
import logging
import subprocess
from dataclasses import dataclass, field

# Shared cell normalization/escaping — single source of truth for all
# pipe-table-producing parsers.
from parsers._tableutil import escape_cell as _cell_to_str
from parsers._tableutil import normalize_cell as _cell_text

logger = logging.getLogger(__name__)

# --- opendataloader-pdf (ODL) support ---

_odl_available_cache = None
_odl_system_python = None   # set when ODL is only accessible via a system Python


def _find_system_python_with_odl():
    """
    Search for a system Python that has opendataloader_pdf and Java installed.
    Used when running inside a Briefcase bundle whose isolated Python cannot
    directly import packages installed in the system environment.

    Returns the executable path of the first qualifying Python, or None.
    """
    # Prefer explicit paths over PATH lookup to avoid accidentally finding
    # the bundled Python itself.
    candidates = []
    if sys.platform == "darwin":
        candidates = [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            os.path.expanduser("~/miniforge3/bin/python3"),
            os.path.expanduser("~/anaconda3/bin/python3"),
            os.path.expanduser("~/.pyenv/shims/python3"),
        ]
    elif sys.platform == "win32":
        # `py` is the Windows Python Launcher — separate from any bundle
        candidates = ["py", "python"]
    else:
        candidates = ["/usr/bin/python3", "/usr/local/bin/python3"]

    bundle_exe = os.path.realpath(sys.executable)

    for candidate in candidates:
        try:
            # Skip if this is the bundle Python
            if os.path.realpath(candidate) == bundle_exe:
                continue
            result = subprocess.run(
                [candidate, "-c", "import opendataloader_pdf; print('ok')"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip() == "ok":
                # Also verify Java is reachable (ODL calls Java internally)
                subprocess.run(["java", "-version"], capture_output=True, timeout=5)
                logger.debug(f"ODL accessible via system Python: {candidate}")
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue
    return None


def odl_available():
    """
    Check if opendataloader-pdf and Java are both available.

    Tries two paths:
    1. Direct import (CLI / development: ODL in the same Python environment).
    2. Subprocess via a system Python (App bundle: ODL installed outside the bundle).

    Result is cached for the session lifetime.
    """
    global _odl_available_cache, _odl_system_python
    if _odl_available_cache is not None:
        return _odl_available_cache

    # Path 1: direct import (fast, same environment)
    try:
        import opendataloader_pdf  # noqa: F401
        subprocess.run(["java", "-version"], capture_output=True, timeout=5)
        _odl_available_cache = True
        return True
    except (ImportError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Path 2: look for a system Python that has ODL (App bundle scenario)
    py = _find_system_python_with_odl()
    if py:
        _odl_system_python = py
        _odl_available_cache = True
    else:
        logger.debug("ODL unavailable (missing opendataloader-pdf or Java)")
        _odl_available_cache = False

    return _odl_available_cache


def clean_odl_output(text):
    """
    Post-process opendataloader-pdf output:
    - Replace <br> / <br/> / <br /> with a single space
    - Remove ![image N](...) lines
    - Compress 3+ consecutive blank lines to 1
    - Strip leading/trailing whitespace
    """
    # ODL emits <br> inside table cells; downstream Markdown consumers (and
    # the in-app preview) treat them as literal text.
    text = re.sub(r"<br\s*/?>", " ", text)
    # Remove image reference lines
    text = re.sub(r"^!\[image \d+\]\([^)]*\)\s*$", "", text, flags=re.MULTILINE)
    # Compress 3+ blank lines to 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _collect_odl_output(filepath):
    """Read and clean up the .md + _images/ side-effects ODL writes beside the input."""
    stem = os.path.splitext(filepath)[0]
    md_path = stem + ".md"

    if not os.path.exists(md_path):
        logger.debug(f"ODL produced no output file for {os.path.basename(filepath)}")
        return None

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        os.remove(md_path)
    except OSError:
        pass
    odl_images_dir = stem + "_images"
    if os.path.isdir(odl_images_dir):
        shutil.rmtree(odl_images_dir, ignore_errors=True)

    return clean_odl_output(text) if text.strip() else None


def extract_text_odl(filepath):
    """
    Extract text from a PDF using opendataloader-pdf (high-quality, table-aware).

    Tries direct import first (CLI), then subprocess via a system Python (App bundle).
    Returns Markdown string on success, None on failure.
    """
    if not odl_available():
        return None

    # --- Direct import path (CLI / same Python environment) ---
    if _odl_system_python is None:
        try:
            from opendataloader_pdf import convert
            convert(filepath, format="markdown")
            return _collect_odl_output(filepath)
        except Exception as e:
            logger.debug(f"ODL direct extraction failed: {e}")
            return None

    # --- Subprocess path (App bundle: call system Python that has ODL) ---
    # Pass filepath via environment variable to avoid any shell-injection risk.
    odl_script = (
        "import os, sys\n"
        "fp = os.environ.get('_ODL_FILE', '')\n"
        "if not fp: sys.exit(1)\n"
        "from opendataloader_pdf import convert\n"
        "convert(fp, format='markdown')\n"
    )
    try:
        result = subprocess.run(
            [_odl_system_python, "-c", odl_script],
            env={**os.environ, "_ODL_FILE": filepath},
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.debug(
                f"ODL subprocess failed: {result.stderr.decode(errors='replace')[:200]}"
            )
            return None
        return _collect_odl_output(filepath)
    except subprocess.TimeoutExpired:
        logger.warning(
            f"ODL subprocess timed out for {os.path.basename(filepath)}"
        )
        return None
    except Exception as e:
        logger.debug(f"ODL subprocess extraction failed: {e}")
        return None

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pikepdf
except ImportError:
    pikepdf = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


def decrypt_pdf(filepath, password=None, output_dir=None):
    """
    Decrypt a password-protected PDF using pikepdf.

    Returns the path to the decrypted file, or None on failure.
    If pikepdf is not installed, the encrypted PDF is returned as-is with a warning.
    """
    if not pikepdf:
        logger.warning("pikepdf not installed — skipping PDF decryption")
        return None
    if not password:
        return None

    filename = os.path.basename(filepath)
    stem, ext = os.path.splitext(filename)
    out_dir = output_dir or os.path.dirname(filepath)
    # When no output_dir specified, add suffix to avoid overwriting the original
    out_name = filename if output_dir else f"{stem}_decrypted{ext}"
    output_path = os.path.join(out_dir, out_name)

    if output_dir and os.path.exists(output_path):
        return output_path

    try:
        with pikepdf.open(filepath, password=password) as pdf:
            os.makedirs(out_dir, exist_ok=True)
            pdf.save(output_path)
        logger.info(f"Decrypted: {filename}")
        return output_path
    except Exception as e:
        logger.error(f"Decryption failed for {filename}: {e}")
        return None


def _optimize_image(image, max_dim=1600):
    """Resize and convert image to RGB, capping at max_dim pixels."""
    from PIL import Image
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return image


def extract_images(filepath, dpi=200, max_pages=15):
    """
    Convert PDF pages to PIL images for AI vision mode.

    Requires: pdf2image + poppler system dependency.
    Returns an empty list if pdf2image is not installed.

    Safety: capped at max_pages to prevent OOM on low-memory machines
    (e.g. Oracle ARM with 1GB RAM). A 200 DPI A4 page ≈ 30MB in memory.
    """
    if not convert_from_path:
        logger.warning("pdf2image not installed — cannot extract PDF images for vision mode")
        return []
    try:
        page_count = get_page_count(filepath)
        if page_count > max_pages:
            logger.warning(
                f"PDF has {page_count} pages, capping vision at {max_pages} to prevent OOM"
            )
        pil_images = convert_from_path(
            filepath, dpi=dpi,
            first_page=1, last_page=min(page_count, max_pages),
        )
        return [_optimize_image(img) for img in pil_images]
    except MemoryError:
        logger.error("OOM during PDF image extraction — try lowering DPI or max_pages")
        return []
    except Exception as e:
        msg = str(e).lower()
        if "poppler" in msg or "pdftoppm" in msg or "pdfinfo" in msg:
            logger.error(
                "poppler not found — required for PDF vision mode.\n"
                "  macOS:  brew install poppler\n"
                "  Ubuntu: sudo apt-get install poppler-utils\n"
                "  Or skip vision: --ai none"
            )
        else:
            logger.error(f"PDF image extraction failed: {e}")
        return []


@dataclass
class _TextPart:
    """A non-table text block on a page, positioned by its top y-coordinate."""
    y: float
    text: str


@dataclass
class _TablePart:
    """A detected table: resolved header cells (str only) + data rows.

    ``header`` never contains None — span-covered header cells resolve to "".
    ``rows`` may contain None for span-covered data cells (rendered blank).
    ``x0``/``x1``/``y1`` carry the table bbox for fragment stitching; the
    zero defaults mean "no geometry" and such parts are never stitched.
    """
    y: float
    header: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    x0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0


def _table_to_markdown(rows, external_header=None):
    """
    Render extracted table cells as a GFM pipe table.

    ``rows`` is ``Table.extract()`` output: list of rows, cells are str or
    None (None = covered by a merged cell → rendered as a blank cell, so the
    merged text appears only in the first cell of its span). The header is
    the first row, unless ``external_header`` is given (PyMuPDF external
    header), in which case every row is data.

    Ragged rows are padded to the widest row; empty cells render as a single
    space to keep cell boundaries visible. Returns "" for empty input.
    An empty ``external_header`` is treated as absent (first row promotes).
    """
    if external_header:
        header, data = list(external_header), rows
    elif rows:
        header, data = rows[0], rows[1:]
    else:
        return ""

    width = max([len(header)] + [len(r) for r in data])
    if width == 0:
        return ""

    def render_row(cells):
        cells = [_cell_to_str(c) for c in cells]
        cells += [""] * (width - len(cells))
        return "|" + "|".join(c if c else " " for c in cells) + "|"

    header_line = render_row(header)
    if not header_line.replace("|", "").strip() and not data:
        return ""

    lines = [header_line, "|" + "|".join(["---"] * width) + "|"]
    lines += [render_row(r) for r in data]
    return "\n".join(lines)


# Fragment stitching thresholds, pinned to real-statement measurements
# (see change stitch-table-fragments notes): intra-table x drift ≤ 0.4 pt,
# fragment gaps 12.6–25.2 pt; nearest false-merge candidates sit at gap
# 14.3 pt (blocked by the multi-row test) and 5.5 pt (blocked by column count).
_STITCH_X_TOL = 6.0
_STITCH_GAP_MAX = 30.0


def _stitch_page_fragments(parts):
    """
    Consolidate same-page single-row table fragments into their preceding
    compatible table.

    find_tables sometimes boxes every row of a statement table individually;
    each such fragment surfaces as a degenerate _TablePart whose only row sits
    in the header position (rows == []). A fragment B stitches into the
    immediately preceding table A iff: no text part lies between them, both
    have the same column count, max(|Δx0|, |Δx1|) <= _STITCH_X_TOL, the
    vertical gap (B.y - A.y1) <= _STITCH_GAP_MAX, and B is degenerate.
    B's header row becomes a data row of A; the chain extends (later
    fragments compare against the stitched result). Pure: inputs are never
    mutated — stitched tables are new _TablePart instances.
    """
    out = []
    prev_idx = None   # index in `out` of the directly preceding table part
    for part in parts:
        if not isinstance(part, _TablePart):
            out.append(part)
            prev_idx = None   # real text breaks the chain
            continue
        if prev_idx is not None:
            a = out[prev_idx]
            stitchable = (
                len(part.rows) == 0
                and len(part.header) == len(a.header)
                and a.x1 > 0.0 and part.x1 > 0.0   # both carry real geometry
                and max(abs(part.x0 - a.x0), abs(part.x1 - a.x1)) <= _STITCH_X_TOL
                and (part.y - a.y1) <= _STITCH_GAP_MAX
            )
            if stitchable:
                out[prev_idx] = _TablePart(
                    a.y, a.header, list(a.rows) + [list(part.header)],
                    x0=a.x0, x1=a.x1, y1=part.y1)
                continue
        out.append(part)
        prev_idx = len(out) - 1
    return out


def _merge_cross_page_tables(pages):
    """
    Flatten per-page part lists, merging cross-page table continuations.

    A table that continues onto the next page (repeated-header convention)
    is detected as: the FIRST table of page N has a header equal — same cell
    texts AND same column count — to the LAST table of page N-1. Its data
    rows are appended to that table (single header in the output); if the
    boundary data row was detected on both pages, it is kept only once.

    A page with no tables breaks any continuation chain. Pure function:
    input parts are never mutated — merged tables are new _TablePart
    instances, so calling twice on the same input gives the same result.
    """
    merged = []
    prev_idx = None   # index in `merged` of the previous page's last table
    for page_parts in pages:
        page_last_idx = None   # None ⇔ no table seen yet on this page
        for part in page_parts:
            if not isinstance(part, _TablePart):
                merged.append(part)
                continue
            if (page_last_idx is None and prev_idx is not None
                    and part.header == merged[prev_idx].header):
                # Continuation: absorb into the previous page's table
                target = merged[prev_idx]
                rows = part.rows
                if rows == target.rows:
                    # Same table fully re-detected on both pages — emit once
                    rows = []
                elif rows and target.rows and rows[0] == target.rows[-1]:
                    rows = rows[1:]
                merged[prev_idx] = _TablePart(
                    target.y, target.header, list(target.rows) + list(rows))
                page_last_idx = prev_idx
            else:
                merged.append(part)
                page_last_idx = len(merged) - 1
        prev_idx = page_last_idx
    return merged


def extract_text_with_tables(filepath):
    """
    Extract PDF text with table detection, preserving table structure as
    Markdown pipe tables (rendered via _table_to_markdown, so merged cells
    stay blank after their first cell) and merging cross-page continuations.

    Returns None on any document-level error so the caller falls back to
    plain text extraction.
    """
    if not fitz:
        return None
    try:
        pages = []
        with fitz.open(filepath) as doc:
            for page in doc:
                pages.append(_stitch_page_fragments(_extract_page_text_with_tables(page)))
        rendered = []
        for part in _merge_cross_page_tables(pages):
            if isinstance(part, _TablePart):
                md = _table_to_markdown(part.rows, external_header=part.header)
            else:
                md = part.text.strip()
            if md:
                rendered.append(md)
        result = "\n\n".join(rendered)
        return result or None
    except Exception as e:
        logger.debug(f"Table-aware extraction failed, falling back to plain text: {e}")
        return None


def _extract_page_text_with_tables(page):
    """Extract one page as y-sorted parts: _TextPart and _TablePart items."""
    def plain():
        text = page.get_text().strip()
        return [_TextPart(0.0, text)] if text else []

    try:
        finder = page.find_tables()
        if not finder.tables:
            return plain()

        parts = []
        table_rects = []
        for tbl in finder.tables:
            header_obj = tbl.header
            if header_obj is None:   # zero-row table: nothing to render
                continue
            extracted = tbl.extract() or []
            if header_obj.external and header_obj.names:
                header, data = header_obj.names, extracted
            elif extracted:
                header, data = extracted[0], extracted[1:]
            else:
                continue   # nothing extractable — leave the region's text alone
            bx0, by0, bx1, by1 = tbl.bbox
            parts.append(_TablePart(by0, [_cell_text(c) for c in header], data,
                                    x0=bx0, x1=bx1, y1=by1))
            # Suppress overlapping text blocks only for tables actually emitted
            table_rects.append(fitz.Rect(tbl.bbox))
            if header_obj.external:
                # External header text sits outside tbl.bbox; exclude it too
                # so it is not emitted again as a stray text block.
                table_rects.append(fitz.Rect(header_obj.bbox))

        # Text blocks, excluding those that overlap significantly with tables
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            if not text.strip():
                continue
            block_rect = fitz.Rect(x0, y0, x1, y1)
            block_area = block_rect.get_area()
            in_table = block_area > 0 and any(
                (block_rect & trect).get_area() / block_area > 0.3
                for trect in table_rects
            )
            if not in_table:
                parts.append(_TextPart(y0, text.strip()))

        parts.sort(key=lambda p: p.y)
        return parts

    except Exception:
        return plain()


def has_tables(filepath):
    """
    First-party routing signal: True if any page has a detectable table.

    Early-exits on the first hit. Never raises — any failure (fitz missing,
    unreadable/corrupt file) returns False so the caller falls back to the
    ODL-first flow (fail-open).
    """
    if not fitz:
        return False
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                if page.find_tables().tables:
                    return True
        return False
    except Exception:
        return False


def get_page_count(filepath):
    """Return the number of pages in a PDF."""
    if not fitz:
        return 1
    try:
        with fitz.open(filepath) as doc:
            n = doc.page_count
        return n
    except Exception:
        return 1
