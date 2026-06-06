"""
EPUB e-book parser — self-parsed ZIP + OPF + XHTML (no external EPUB library).

EbookLib is deliberately avoided: it is AGPL-licensed, incompatible with this
project's MIT license and binary distribution. EPUB structure is simple enough
to parse directly with stdlib `zipfile` + `lxml` (already a dependency):

  META-INF/container.xml  → declares the OPF package path
  OPF <manifest>          → id → href map of every resource
  OPF <spine>             → ordered itemrefs = reading order (EPUB 2 and 3)
  each spine XHTML         → one chapter

All XML is namespaced (container, OPF/IDPF, Dublin Core), so every lookup uses
`local-name()` XPath rather than a hard-coded prefix/nsmap. hrefs are fragment-
stripped and percent-decoded before resolving against the OPF directory. Output
is `# <title>` / author head + one `## <heading>` section per chapter, each
passed through the shared noise cleaner (consistency with the .pdf/iWork paths).
"""
import logging
import os
import posixpath
import urllib.parse
import zipfile

from lxml import etree
from lxml import html as lxml_html

from classifiers.noise import clean_text

logger = logging.getLogger(__name__)

MAX_DECOMPRESSED_SIZE = 500 * 1024 * 1024  # 500MB, mirrors parsers.pptx
OBJECT_REPLACEMENT = "￼"  # ￼ placeholder for images/embeds in the XHTML
CONTAINER_PATH = "META-INF/container.xml"
ENCRYPTION_PATH = "META-INF/encryption.xml"
# block-level tags whose text becomes a paragraph
_BLOCK_TAGS = ("p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6",
               "li", "blockquote", "pre", "td", "th", "dd", "dt", "figcaption")


def parse(filepath: str) -> str:
    """Extract chapter text from an .epub in spine reading order.

    Returns the cleaned Markdown string, or "" on any failure (oversize, DRM,
    corrupt/missing container or OPF, pipeline error). No parse-level exception
    propagates to the caller.
    """
    name = os.path.basename(filepath)
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            names = set(zf.namelist())

            # D6 — ZIP-bomb guard: declared-size sweep before reading
            total = sum(i.file_size for i in zf.infolist())
            if total > MAX_DECOMPRESSED_SIZE:
                logger.warning(
                    f"EPUB decompressed size ({total / 1024 / 1024:.0f}MB) "
                    f"exceeds {MAX_DECOMPRESSED_SIZE / 1024 / 1024:.0f}MB in {name}"
                )
                return ""

            # D7 — DRM: reject only if content (OPF/XHTML) is encrypted
            if ENCRYPTION_PATH in names and _content_is_encrypted(zf):
                logger.warning(
                    f"{name} is DRM-protected (encrypted content); not supported"
                )
                return ""

            opf_path = _locate_opf(zf, names)
            if not opf_path or opf_path not in names:
                logger.warning(f"No readable OPF package found in {name}")
                return ""

            opf = etree.fromstring(zf.read(opf_path))
            opf_dir = posixpath.dirname(opf_path)

            title, creator = _read_metadata(opf)
            manifest = _read_manifest(opf)            # id -> href
            spine = _read_spine(opf)                   # [(idref, linear_bool)]
            nav_ids = _nav_ids(opf)                    # EPUB3 nav/TOC docs to skip

            sections = _read_chapters(
                zf, names, spine, manifest, nav_ids, opf_dir, name
            )
            if not sections:
                logger.warning(f"No chapter text extracted from {name}")
                return ""

            head = []
            if title:
                head.append(f"# {title}")
            if creator:
                head.append(f"作者：{creator}")
            head_str = ("\n".join(head) + "\n\n") if head else ""
            return head_str + "\n\n".join(sections)

    except zipfile.BadZipFile as e:
        logger.warning(f"Not a valid EPUB ZIP archive {name}: {e}")
        return ""
    except Exception as e:
        logger.warning(f"EPUB parse failed for {name}: {e}")
        return ""


# ── DRM / encryption ───────────────────────────────────────────────────────────

def _content_is_encrypted(zf):
    """True only if encryption.xml encrypts the OPF or an XHTML content doc.

    Font-obfuscation (the other use of encryption.xml) encrypts only font files,
    leaving content plaintext — those EPUBs extract fine, so are not rejected.
    """
    try:
        enc = etree.fromstring(zf.read(ENCRYPTION_PATH))
    except Exception:
        return False
    uris = enc.xpath("//*[local-name()='CipherReference']/@URI")
    for uri in uris:
        path = urllib.parse.unquote(uri.split("#", 1)[0]).lower()
        if path.endswith((".opf", ".xhtml", ".html", ".htm")):
            return True
    return False


# ── container / OPF location ─────────────────────────────────────────────────────

def _locate_opf(zf, names):
    """Read META-INF/container.xml and return the declared OPF rootfile path."""
    if CONTAINER_PATH not in names:
        return None
    try:
        container = etree.fromstring(zf.read(CONTAINER_PATH))
    except Exception:
        return None
    paths = container.xpath("//*[local-name()='rootfile']/@full-path")
    return paths[0] if paths else None


# ── OPF parsing (namespace-agnostic via local-name) ─────────────────────────────

def _read_metadata(opf):
    """Return (title, creator) from OPF Dublin Core metadata; '' if absent."""
    def first(tag):
        vals = opf.xpath(
            f"//*[local-name()='metadata']/*[local-name()=$t]/text()", t=tag
        )
        return vals[0].strip() if vals else ""
    return first("title"), first("creator")


def _read_manifest(opf):
    """Return {id: href} from the OPF manifest."""
    out = {}
    for item in opf.xpath("//*[local-name()='manifest']/*[local-name()='item']"):
        item_id = item.get("id")
        href = item.get("href")
        if item_id and href:
            out[item_id] = href
    return out


def _read_spine(opf):
    """Return ordered [(idref, linear)] from the OPF spine."""
    spine = []
    for ref in opf.xpath("//*[local-name()='spine']/*[local-name()='itemref']"):
        idref = ref.get("idref")
        if idref:
            spine.append((idref, (ref.get("linear", "yes") or "").strip().lower() != "no"))
    return spine


def _nav_ids(opf):
    """IDs of EPUB 3 navigation documents (manifest item properties='nav').

    The nav/TOC document is in the spine reading order but is a table of contents,
    not prose — extracting it would dump the whole TOC into the output.
    """
    ids = set()
    for item in opf.xpath("//*[local-name()='manifest']/*[local-name()='item']"):
        if "nav" in (item.get("properties") or "").split() and item.get("id"):
            ids.add(item.get("id"))
    return ids


# ── chapter extraction ──────────────────────────────────────────────────────────

def _read_chapters(zf, names, spine, manifest, nav_ids, opf_dir, name):
    """Walk linear spine items in order, returning `## heading\\n\\ntext` sections."""
    sections = []
    chapter_no = 0
    for idref, linear in spine:
        if not linear or idref in nav_ids:
            continue  # skip non-linear items and EPUB3 nav/TOC documents
        chapter_no += 1  # consume a number even if this item is skipped/empty

        href = manifest.get(idref)
        if not href:
            logger.debug(f"{name}: spine idref '{idref}' not in manifest; skipping")
            continue
        entry = _resolve_href(opf_dir, href)
        if entry not in names:
            logger.debug(f"{name}: resolved entry '{entry}' missing; skipping")
            continue

        try:
            heading, body = _xhtml_to_text(zf.read(entry))
        except Exception as e:
            logger.debug(f"{name}: failed to read chapter {entry}: {e}")
            continue
        body = clean_text(body)
        if not body.strip():
            continue
        sections.append(f"## {heading or f'章節 {chapter_no}'}\n\n{body}")
    return sections


def _resolve_href(opf_dir, href):
    """Resolve a manifest href to a ZIP entry path: strip fragment, decode, join."""
    path = urllib.parse.unquote(href.split("#", 1)[0])
    joined = posixpath.join(opf_dir, path) if opf_dir else path
    return posixpath.normpath(joined)


def _xhtml_to_text(xhtml_bytes):
    """Return (heading, body_text) from an XHTML chapter via lxml.html.

    EPUB content documents are UTF-8 (per spec); force the parser to UTF-8 so
    libxml2's HTML default of latin-1 doesn't mojibake CJK text.
    """
    parser = lxml_html.HTMLParser(encoding="utf-8")
    doc = lxml_html.fromstring(xhtml_bytes, parser=parser)
    for bad in doc.xpath("//*[local-name()='script' or local-name()='style']"):
        bad.getparent().remove(bad)

    # heading: first non-empty h1/h2 element (tracked so we don't repeat it in
    # the body), else <title>
    heading = ""
    heading_el = None
    # any heading level (h1-h6): some books title chapters with h3 etc. and put
    # the (repeated) book name in <title>, so prefer a real heading element over
    # the <title> fallback to avoid every chapter sharing the book-name heading
    for tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        for el in doc.xpath(f"//*[local-name()='{tag}']"):
            t = " ".join(el.text_content().replace(OBJECT_REPLACEMENT, " ").split())
            if t:
                heading, heading_el = t, el
                break
        if heading_el is not None:
            break
    if not heading:
        ttext = doc.xpath("//*[local-name()='title']//text()")
        if ttext:
            heading = " ".join(" ".join(ttext).replace(OBJECT_REPLACEMENT, " ").split())

    # body: text of block-level elements (one paragraph each), skipping the
    # element already used as the heading
    cond = " or ".join(f"local-name()='{t}'" for t in _BLOCK_TAGS)
    parts = []
    # leaf blocks only — a block with no block-level descendant. This avoids
    # double-counting nested blocks (blockquote>p, div>p, li>p) whose parent and
    # child carry the same text_content, and lets div/section-wrapped prose be
    # captured (the wrapping div is non-leaf, its innermost block is the leaf).
    for b in doc.xpath(f"//*[{cond}][not(.//*[{cond}])]"):
        if b is heading_el:
            continue
        txt = " ".join(b.text_content().replace(OBJECT_REPLACEMENT, " ").split())
        # drop blocks identical to the heading — cover/title/half-title pages
        # repeat the book/chapter title as both heading and body
        if txt and txt != heading:
            parts.append(txt)
    if not parts and heading_el is None:  # no headings, no leaf blocks
        # fall back to <body> text only (never <head>/<title>)
        body_els = doc.xpath("//*[local-name()='body']")
        src = body_els[0] if body_els else doc
        whole = " ".join(src.text_content().replace(OBJECT_REPLACEMENT, " ").split())
        if whole and whole != heading:
            parts.append(whole)
    return heading, "\n\n".join(parts)
