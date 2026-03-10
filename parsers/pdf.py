"""
PDF parser — decrypt + native text extraction + image extraction for vision mode.

Supports:
- Decryption via pikepdf (optional, graceful skip)
- Native text extraction via PyMuPDF (fitz)
- Image extraction via pdf2image (optional, for AI vision mode)
- Image optimization (RGB, max 1600px) to save tokens
"""
import os
import logging

logger = logging.getLogger(__name__)

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


def extract_text(filepath, max_chars=150_000):
    """
    Extract native text from PDF using PyMuPDF (fitz).

    Returns raw text string. Returns empty string if fitz is not installed
    or the PDF has no extractable text layer.
    """
    if not fitz:
        logger.warning("PyMuPDF (fitz) not installed — cannot extract PDF text")
        return ""
    try:
        with fitz.open(filepath) as doc:
            parts = []
            for page in doc:
                parts.append(page.get_text())
                if sum(len(p) for p in parts) > max_chars:
                    break
        return "".join(parts)[:max_chars]
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return ""


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
