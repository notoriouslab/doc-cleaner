"""
Apple Pages (.pages) and Keynote (.key) parser — multi-strategy.

Modern iWork (2020+, format v14.x) stores content as `Index/*.iwa` (Apple's
IWA: Protobuf + Snappy) and embeds only a single-slide `preview.jpg`, NOT a
`QuickLook/Preview.pdf`. Legacy iWork '09/early-'13 files embed a text-native
`QuickLook/Preview.pdf` instead.

Strategies:
- `.key`: decode IWA via `keynote-parser` and emit one `## 投影片 N` section per
  slide in presentation order (primary). If that yields nothing, fall back to an
  embedded `QuickLook/Preview.pdf` (legacy files).
- `.pages`: extract from `QuickLook/Preview.pdf` (legacy files). No maintained
  IWA library exists for Pages, so modern files return "" with an instruction to
  export to PDF.

All paths return "" on failure; no parse-level exception propagates (a temp-file
write error is a system failure and is allowed to propagate). A 500 MB ZIP-bomb
guard protects both the IWA pre-decode sweep and the QuickLook-PDF entry read.
"""
import os

# protobuf descriptor-pool guard (D10): numbers-parser and keynote-parser both
# vendor the same Apple .proto names (TSP/TSK/TSS/TSDArchives). The C/upb
# protobuf implementation aborts with "duplicate file name" when both are
# imported in one process (e.g. a batch with .numbers and .key); the pure-Python
# implementation tolerates it. Must be set before any protobuf import.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import logging
import tempfile
import zipfile

from classifiers.noise import clean_text
from classifiers.pdf_classifier import PdfType, classify
from parsers import pdf

logger = logging.getLogger(__name__)

QUICKLOOK_PDF = "QuickLook/Preview.pdf"
MAX_DECOMPRESSED_SIZE = 500 * 1024 * 1024  # 500MB, mirrors parsers.pptx
OBJECT_REPLACEMENT = "￼"  # ￼ placeholder Keynote uses for images/embeds


def parse(filepath: str) -> str:
    """Extract text from a .pages/.key file. Dispatches by extension."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".key":
        return _parse_key(filepath)
    # .pages (and any other extension routed here) uses the QuickLook-PDF path
    return _parse_pages(filepath)


# ── Keynote (.key): IWA primary, QuickLook-PDF fallback ────────────────────────

def _parse_key(filepath: str) -> str:
    """IWA decoding (primary) → QuickLook-PDF fallback → "" with warning."""
    name = os.path.basename(filepath)
    if _within_size_limit(filepath, name):
        iwa_text = _extract_iwa_slides(filepath, name)
        if iwa_text:
            return iwa_text
    # Fallback: legacy .key with an embedded QuickLook PDF.
    pdf_text = _extract_quicklook_pdf(filepath)
    if pdf_text:
        return pdf_text
    logger.warning(
        f"No extractable content in {name} (no IWA slide text and no QuickLook PDF)"
    )
    return ""


def _extract_iwa_slides(filepath: str, name: str) -> str:
    """Decode Index/*.iwa and return slide text as `## 投影片 N` sections.

    Returns "" if keynote-parser is unavailable or decoding yields nothing, so
    the caller can fall back. Uses IWAFile.from_buffer (no file_reader, hence no
    tqdm progress bar — critical under the tty-less GUI).
    """
    try:
        from keynote_parser.codec import IWAFile
    except ImportError:
        logger.warning(
            f"keynote-parser not installed; cannot decode IWA for {name}"
        )
        return ""

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            names = set(zf.namelist())
            store = _build_object_store(zf, IWAFile, names)
            order = _slide_proxy_order(zf, IWAFile, names)

            sections = []
            if order:
                # slideTree gives presentation order as proxy object IDs; each
                # proxy's objectReferences[0] points at the real Slide-<id>.iwa.
                for n, proxy_id in enumerate(order, 1):
                    file_id = _proxy_to_file_id(store, proxy_id)
                    if file_id is None:
                        continue
                    text = _slide_file_text(zf, IWAFile, file_id, names)
                    if text:
                        sections.append(f"## 投影片 {n}\n\n{text}")
            else:
                logger.warning(
                    f"Could not read slideTree in {name}; ordering slides by "
                    f"object ID (best-effort, may not match presentation order)"
                )
                slide_files = sorted(
                    (x for x in names if _slide_file_id(x) is not None),
                    key=lambda x: int(_slide_file_id(x)),
                )
                for n, fn in enumerate(slide_files, 1):
                    text = _slide_file_text(zf, IWAFile, _slide_file_id(fn), names)
                    if text:
                        sections.append(f"## 投影片 {n}\n\n{text}")

            return "\n\n".join(sections)
    except zipfile.BadZipFile as e:
        logger.warning(f"Not a valid Keynote ZIP archive {name}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"IWA decode failed for {name}: {e}")
        return ""


def _build_object_store(zf, IWAFile, names):
    """Map every IWA object's identifier → its archive dict, across all Index/*.iwa.

    The slideTree references slide-proxy objects that live in Document.iwa, not in
    the Slide-*.iwa files, so a global store is needed to resolve them.
    """
    store = {}
    for n in names:
        if not _is_index_iwa(n):
            continue
        try:
            d = IWAFile.from_buffer(zf.read(n), n).to_dict()
        except Exception:
            continue
        for chunk in d.get("chunks", []):
            for arch in chunk.get("archives", []):
                oid = arch.get("header", {}).get("identifier")
                if oid is not None:
                    store[str(oid)] = arch
    return store


def _slide_proxy_order(zf, IWAFile, names):
    """Return the ordered list of slide-proxy IDs from Document.iwa's slideTree.

    Returns None if it cannot be read (caller falls back to filename-ID order).
    """
    if "Index/Document.iwa" not in names:
        return None
    try:
        d = IWAFile.from_buffer(
            zf.read("Index/Document.iwa"), "Index/Document.iwa"
        ).to_dict()
    except Exception:
        return None
    tree = _find_first_key(d, "slideTree")
    if not isinstance(tree, dict):
        return None
    slides = tree.get("slides")
    if not isinstance(slides, list):
        return None
    ids = [str(s["identifier"]) for s in slides
           if isinstance(s, dict) and "identifier" in s]
    return ids or None


def _proxy_to_file_id(store, proxy_id):
    """Resolve a slide-proxy ID to its referenced Slide-<id>.iwa file ID."""
    arch = store.get(str(proxy_id))
    if not arch:
        return None
    for mi in arch.get("header", {}).get("messageInfos", []):
        refs = mi.get("objectReferences")
        if refs:
            return str(refs[0])
    return None


def _slide_file_text(zf, IWAFile, file_id, names):
    """Decode Index/Slide-<file_id>.iwa and return its joined text storage."""
    fn = f"Index/Slide-{file_id}.iwa"
    if fn not in names:
        return ""
    try:
        d = IWAFile.from_buffer(zf.read(fn), fn).to_dict()
    except Exception:
        return ""
    hits = []
    _collect_text(d, hits)
    return "\n".join(hits)


def _collect_text(obj, hits):
    """Recursively gather TSWP text-storage strings, stripping ￼ placeholders.

    TSWP text storage is always a `list[str]` in the IWA schema (verified on real
    files), so only list-valued `text` keys are collected; any other shape is
    walked through as an ordinary node.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text" and isinstance(v, list):
                for s in v:
                    if isinstance(s, str):
                        cleaned = s.replace(OBJECT_REPLACEMENT, "").strip()
                        if cleaned:
                            hits.append(cleaned)
            else:
                _collect_text(v, hits)
    elif isinstance(obj, list):
        for x in obj:
            _collect_text(x, hits)


def _find_first_key(obj, target):
    """Depth-first search for the first value under key `target`."""
    if isinstance(obj, dict):
        if target in obj:
            return obj[target]
        for v in obj.values():
            r = _find_first_key(v, target)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for x in obj:
            r = _find_first_key(x, target)
            if r is not None:
                return r
    return None


def _is_index_iwa(name):
    return name.startswith("Index/") and name.endswith(".iwa")


def _slide_file_id(name):
    """Return the numeric ID of an Index/Slide-<id>.iwa file, or None."""
    if not (name.startswith("Index/Slide-") and name.endswith(".iwa")):
        return None
    stem = name[len("Index/Slide-"):-len(".iwa")]
    return stem if stem.isdigit() else None


def _within_size_limit(filepath, name):
    """ZIP-bomb guard for the IWA path (D9): sum declared uncompressed sizes.

    Mirrors parsers.pptx._check_zip_size. Returns False (and warns) if the total
    exceeds the limit, so the caller skips IWA decoding. A non-ZIP file returns
    True — the downstream open will surface the real error.
    """
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            total = sum(i.file_size for i in zf.infolist())
    except Exception:
        return True
    if total > MAX_DECOMPRESSED_SIZE:
        logger.warning(
            f"{name} decompressed size ({total / 1024 / 1024:.0f}MB) exceeds "
            f"{MAX_DECOMPRESSED_SIZE / 1024 / 1024:.0f}MB; skipping IWA decode"
        )
        return False
    return True


# ── Pages (.pages): QuickLook-PDF only, else export-to-PDF instruction ──────────

def _parse_pages(filepath: str) -> str:
    """QuickLook-PDF extraction; modern files (no PDF) get an export instruction."""
    name = os.path.basename(filepath)
    if QUICKLOOK_PDF not in _zip_names(filepath):
        logger.warning(
            f"No extractable content in {name}: modern Pages stores content in "
            f"IWA, not PDF. Open it in Pages and use File → Export → PDF, then "
            f"clean the PDF."
        )
        return ""
    return _extract_quicklook_pdf(filepath)


def _zip_names(filepath):
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            return set(zf.namelist())
    except Exception:
        return set()


# ── Shared QuickLook-PDF path (retained from the original implementation) ───────

def _extract_quicklook_pdf(filepath: str) -> str:
    """Extract the embedded QuickLook/Preview.pdf through the PDF pipeline.

    Returns "" on absent PDF, oversize entry, corrupt ZIP, or pipeline failure.
    Used as the .key fallback and the .pages primary path.
    """
    name = os.path.basename(filepath)
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            try:
                info = zf.getinfo(QUICKLOOK_PDF)
            except KeyError:
                logger.warning(f"No QuickLook/Preview.pdf in {name}")
                return ""
            if info.file_size > MAX_DECOMPRESSED_SIZE:
                logger.warning(
                    f"QuickLook PDF too large in {name} "
                    f"({info.file_size / 1024 / 1024:.0f}MB exceeds "
                    f"{MAX_DECOMPRESSED_SIZE / 1024 / 1024:.0f}MB)"
                )
                return ""
            pdf_bytes = zf.read(QUICKLOOK_PDF)
            # Defense-in-depth: getinfo().file_size is the central-directory
            # declared size, which a crafted archive could understate. Re-check
            # the actual decompressed length before it reaches the pipeline.
            if len(pdf_bytes) > MAX_DECOMPRESSED_SIZE:
                logger.warning(
                    f"QuickLook PDF decompressed too large in {name} "
                    f"({len(pdf_bytes) / 1024 / 1024:.0f}MB exceeds "
                    f"{MAX_DECOMPRESSED_SIZE / 1024 / 1024:.0f}MB)"
                )
                return ""
    except zipfile.BadZipFile as e:
        logger.warning(f"Not a valid iWork ZIP archive {name}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"Failed to open {name}: {e}")
        return ""

    return _extract_pdf_bytes(pdf_bytes, name)


def _extract_pdf_bytes(pdf_bytes: bytes, name: str) -> str:
    """Write PDF bytes to a temp file, run the PDF pipeline, always clean up.

    The temp file is deleted in a finally block on every exit path. A failure
    to even write the temp file (disk full) propagates — it is not a parse
    failure — but the partial temp file is still removed first.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    try:
        tmp.write(pdf_bytes)
    except Exception:
        tmp.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise  # system-level failure, not a parse failure
    finally:
        tmp.close()

    try:
        # Mirrors the .pdf branch in cleaner.parse_file(): NATIVE and
        # LAYOUT_BROKEN both benefit from table recovery; SCANNED never occurs
        # for a text-native QuickLook preview.
        pdf_type, raw_text, _meta = classify(tmp_path)
        if pdf_type in (PdfType.NATIVE, PdfType.LAYOUT_BROKEN):
            table_text = pdf.extract_text_with_tables(tmp_path)
            if table_text:
                raw_text = table_text
        text = clean_text(raw_text)
        if not text.strip():
            logger.warning(f"No text extracted from QuickLook PDF in {name}")
            return ""
        return text
    except Exception as e:
        logger.warning(f"PDF pipeline failed for {name}: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
