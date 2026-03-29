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
import glob
import shutil
import logging
import subprocess

logger = logging.getLogger(__name__)

# --- opendataloader-pdf (ODL) support ---

_odl_available_cache = None


def odl_available():
    """
    Check if opendataloader-pdf and Java are both available.
    Result is cached for the session lifetime.
    """
    global _odl_available_cache
    if _odl_available_cache is not None:
        return _odl_available_cache

    # Check Python package
    try:
        import opendataloader_pdf  # noqa: F401
    except ImportError:
        logger.debug("opendataloader-pdf not installed — ODL extraction unavailable")
        _odl_available_cache = False
        return False

    # Check Java
    try:
        subprocess.run(
            ["java", "-version"],
            capture_output=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Java not available on PATH — ODL extraction unavailable")
        _odl_available_cache = False
        return False

    _odl_available_cache = True
    return True


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


def extract_text_odl(filepath):
    """
    Extract text from a PDF using opendataloader-pdf (Fast mode).

    Returns Markdown string on success, None on failure.
    ODL writes a .md file and *_images/ dir as side effects — both are cleaned up.
    """
    if not odl_available():
        return None

    try:
        from opendataloader_pdf import convert
        convert(filepath, format="markdown")

        # ODL writes output alongside the input file
        stem = os.path.splitext(filepath)[0]
        md_path = stem + ".md"
        images_dir = stem + "_images"

        if not os.path.exists(md_path):
            logger.debug(f"ODL produced no output file for {os.path.basename(filepath)}")
            return None

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Cleanup side-effect files
        try:
            os.remove(md_path)
        except OSError:
            pass
        if os.path.isdir(images_dir):
            try:
                shutil.rmtree(images_dir)
            except OSError:
                pass
        # Also clean up any glob pattern images dirs
        for d in glob.glob(stem + "_images*"):
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d)
                except OSError:
                    pass

        if not text or not text.strip():
            return None

        return clean_odl_output(text)

    except Exception as e:
        logger.debug(f"ODL extraction failed for {os.path.basename(filepath)}: {e}")
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
