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
    - Remove ![image N](...) lines
    - Compress 3+ consecutive blank lines to 1
    - Strip leading/trailing whitespace
    """
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
        logger.error(f"OOM during PDF image extraction — try lowering DPI or max_pages")
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


def extract_text_with_tables(filepath):
    """
    Extract PDF text with table detection, preserving table structure as Markdown.

    Uses PyMuPDF find_tables() to locate tables on each page, converts them to
    Markdown pipe tables via Table.to_markdown(), and interleaves them with the
    surrounding text sorted by vertical position.

    Falls back to plain page.get_text() on any error.
    """
    if not fitz:
        return None
    try:
        parts_all_pages = []
        with fitz.open(filepath) as doc:
            for page in doc:
                parts_all_pages.append(_extract_page_text_with_tables(page))
        result = "\n\n".join(p for p in parts_all_pages if p)
        return result or None
    except Exception as e:
        logger.debug(f"Table-aware extraction failed, falling back to plain text: {e}")
        return None


def _extract_page_text_with_tables(page):
    """Extract one page's text, converting detected tables to Markdown."""
    try:
        finder = page.find_tables()
        if not finder.tables:
            return page.get_text().strip()

        # Build table entries: (y_top, markdown_string)
        table_entries = []
        table_rects = []
        for tbl in finder.tables:
            tbl_rect = fitz.Rect(tbl.bbox)
            table_rects.append(tbl_rect)
            table_entries.append((tbl.bbox[1], tbl.to_markdown()))

        # Get text blocks, excluding those that overlap significantly with tables
        text_entries = []
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
                text_entries.append((y0, text.strip()))

        # Merge text blocks and tables sorted by vertical position
        all_entries = text_entries + table_entries
        all_entries.sort(key=lambda e: e[0])
        return "\n\n".join(content for _, content in all_entries if content.strip())

    except Exception:
        return page.get_text().strip()


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
