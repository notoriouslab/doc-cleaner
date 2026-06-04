"""
Platform-specific utilities for document conversion and file management.

Single source of truth for all platform branches so callers never need to
check platform.system() themselves.

macOS  — textutil (system built-in) for legacy Office; Finder for reveal
Windows — LibreOffice headless for legacy Office (if installed); Explorer for reveal
Linux  — LibreOffice headless; xdg-open parent directory for reveal
"""
import logging
import platform
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM = platform.system()   # "Darwin" | "Windows" | "Linux"


# ── Legacy Office conversion (.doc, .ppt) ────────────────────────────────────

def convert_legacy_office(filepath: str, format_label: str = "file") -> str:
    """
    Convert a legacy Office binary format (.doc, .ppt) to plain text.

    Delegates to the best available converter for the current platform:
      macOS   → /usr/bin/textutil (system built-in, no install needed)
      Windows → LibreOffice --headless (if installed at standard paths)
      Linux   → LibreOffice --headless (if available on PATH)
    """
    if SYSTEM == "Darwin":
        from parsers._textutil import convert_to_text
        return convert_to_text(filepath, format_label=format_label)
    else:
        return _libreoffice_to_text(filepath, format_label=format_label)


def _libreoffice_to_text(filepath: str, format_label: str) -> str:
    """Convert via LibreOffice --headless (Windows / Linux)."""
    soffice = _find_libreoffice()
    if not soffice:
        logger.warning(
            f"{format_label} 轉換需要 LibreOffice（Windows/Linux 平台）。"
            "請至 https://www.libreoffice.org 下載安裝。"
        )
        return ""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "txt:Text",
                 filepath, "--outdir", tmpdir],
                check=True,
                capture_output=True,
                timeout=60,
            )
            out_path = Path(tmpdir) / (Path(filepath).stem + ".txt")
            if out_path.exists():
                text = out_path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    return text
    except subprocess.TimeoutExpired:
        logger.warning(f"LibreOffice {format_label} 轉換超時（60s）")
    except Exception as e:
        logger.warning(f"LibreOffice {format_label} 轉換失敗：{e}")
    return ""


def _find_libreoffice() -> str | None:
    """Return path to soffice/LibreOffice executable, or None if not found."""
    import shutil
    # Standard Windows install locations
    win_paths = [
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ]
    for p in win_paths:
        if Path(p).exists():
            return p
    # PATH lookup (Linux + non-standard Windows)
    return shutil.which("soffice") or shutil.which("libreoffice")


# ── File manager reveal ───────────────────────────────────────────────────────

def reveal_in_file_manager(path: str) -> None:
    """
    Reveal a file in the platform's file manager (non-blocking, fire-and-forget).

    Rejects relative paths and URL schemes as a defence-in-depth measure.

    macOS   → Finder  (open -R)
    Windows → Explorer (explorer /select,)
    Linux   → opens the parent directory (xdg-open)
    """
    if not isinstance(path, str) or not Path(path).is_absolute():
        return
    if "://" in path:
        return

    if SYSTEM == "Darwin":
        subprocess.run(["/usr/bin/open", "-R", path], check=False)
    elif SYSTEM == "Windows":
        # /select, highlights the file; comma+space before path is required
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", str(Path(path).parent)], check=False)
