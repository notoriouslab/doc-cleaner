"""
Shared macOS textutil conversion helper.

Used by docx.py (DOCX fallback + DOC) and pptx.py (PPT).
"""
import os
import logging
import platform
import subprocess
import tempfile

logger = logging.getLogger(__name__)

TEXTUTIL_TIMEOUT = 60  # seconds


def convert_to_text(filepath, format_label="file"):
    """Convert a document to plain text via macOS textutil.

    Args:
        filepath: path to the source file
        format_label: human-readable format name for log messages

    Returns:
        extracted text string, or empty string on failure / non-macOS
    """
    if platform.system() != "Darwin":
        logger.warning(
            f"{format_label.upper()} extraction requires macOS textutil "
            f"(not available on this platform)"
        )
        return ""

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_txt = os.path.join(tmpdir, "out.txt")
            subprocess.run(
                # list args (not shell=True) — safe from shell injection
                # Absolute path: Finder/LaunchServices launch sets a minimal PATH
                ["/usr/bin/textutil", "-convert", "txt", filepath, "-output", temp_txt],
                check=True,
                capture_output=True,
                timeout=TEXTUTIL_TIMEOUT,
            )
            if os.path.exists(temp_txt):
                with open(temp_txt, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if content.strip():
                    return content
    except subprocess.TimeoutExpired:
        logger.warning(f"textutil {format_label} conversion timed out after {TEXTUTIL_TIMEOUT}s")
    except Exception as e:
        logger.warning(f"textutil {format_label} conversion failed: {e}")

    return ""
