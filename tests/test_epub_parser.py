"""Tests for parsers/epub.py — self-parsed EPUB (ZIP + OPF + XHTML).

All fixtures are built in-memory with `zipfile` and carry the real EPUB
namespaces (container, IDPF/OPF, Dublin Core, XHTML) so the parser's
`local-name()` strategy is genuinely exercised. No checked-in binary fixture.
"""
import logging
import os
import zipfile
from unittest.mock import patch

import pytest

from parsers import epub

CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"
OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
XHTML_NS = "http://www.w3.org/1999/xhtml"
ENC_NS = "http://www.w3.org/2001/04/xmlenc#"


def _container(opf_path="OEBPS/content.opf"):
    return (
        f'<?xml version="1.0"?>'
        f'<container version="1.0" xmlns="{CONTAINER_NS}">'
        f'<rootfiles><rootfile full-path="{opf_path}" '
        f'media-type="application/oebps-package+xml"/></rootfiles></container>'
    )


def _opf(manifest, spine, title="測試書", creator="測試作者"):
    """manifest: [(id, href)]; spine: [(idref, linear_bool)]."""
    items = "".join(
        f'<item id="{i}" href="{h}" media-type="application/xhtml+xml"/>'
        for i, h in manifest
    )
    refs = "".join(
        f'<itemref idref="{r}"{"" if lin else " linear=\"no\""}/>'
        for r, lin in spine
    )
    meta = ""
    if title is not None:
        meta += f"<dc:title>{title}</dc:title>"
    if creator is not None:
        meta += f"<dc:creator>{creator}</dc:creator>"
    return (
        f'<?xml version="1.0"?>'
        f'<package xmlns="{OPF_NS}" version="3.0" unique-identifier="id">'
        f'<metadata xmlns:dc="{DC_NS}">{meta}</metadata>'
        f"<manifest>{items}</manifest>"
        f"<spine>{refs}</spine></package>"
    )


def _xhtml(title="", h1="", body="本文段落。"):
    head = f"<title>{title}</title>" if title else ""
    heading = f"<h1>{h1}</h1>" if h1 else ""
    return (
        f'<?xml version="1.0" encoding="utf-8"?>'
        f'<html xmlns="{XHTML_NS}"><head>{head}</head>'
        f"<body>{heading}<p>{body}</p></body></html>"
    )


def _build(tmp_path, *, entries, name="book.epub"):
    """entries: {arcname: str|bytes}. Returns the .epub path."""
    p = tmp_path / name
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        for arc, data in entries.items():
            zf.writestr(arc, data)
    return str(p)


def _standard(tmp_path, **opf_kw):
    """A normal 2-chapter EPUB."""
    entries = {
        "META-INF/container.xml": _container(),
        "OEBPS/content.opf": _opf(
            [("c1", "ch1.xhtml"), ("c2", "ch2.xhtml")],
            [("c1", True), ("c2", True)],
            **opf_kw,
        ),
        "OEBPS/ch1.xhtml": _xhtml(h1="第一章", body="第一章內容"),
        "OEBPS/ch2.xhtml": _xhtml(h1="第二章", body="第二章內容"),
    }
    return _build(tmp_path, entries=entries)


# ── TestSpineOrder ─────────────────────────────────────────────────────────────

class TestSpineOrder:
    def test_chapters_in_spine_order(self, tmp_path):
        result = epub.parse(_standard(tmp_path))
        assert result.count("## ") == 2
        assert result.index("第一章") < result.index("第二章")

    def test_non_linear_skipped(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf(
                [("c1", "ch1.xhtml"), ("c2", "ch2.xhtml")],
                [("c1", True), ("c2", False)],  # c2 linear="no"
            ),
            "OEBPS/ch1.xhtml": _xhtml(h1="第一章", body="主要內容"),
            "OEBPS/ch2.xhtml": _xhtml(h1="附註", body="非線性附註"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "主要內容" in result
        assert "非線性附註" not in result

    def test_missing_idref_skipped_others_extract(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf(
                [("c1", "ch1.xhtml")],  # c2 has no manifest item
                [("c1", True), ("c2", True)],
            ),
            "OEBPS/ch1.xhtml": _xhtml(h1="第一章", body="仍可抽取"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "仍可抽取" in result  # broken c2 ref didn't empty the book

    def test_nav_toc_document_not_extracted(self, tmp_path):
        """EPUB3 nav (TOC) document in the spine must not be extracted as prose."""
        opf = (
            f'<?xml version="1.0"?><package xmlns="{OPF_NS}" version="3.0" '
            f'unique-identifier="id"><metadata xmlns:dc="{DC_NS}">'
            f"<dc:title>書</dc:title></metadata><manifest>"
            f'<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
            f'<item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
            f"</manifest><spine>"
            f'<itemref idref="nav"/><itemref idref="c1"/></spine></package>'
        )
        nav = (
            f'<html xmlns="{XHTML_NS}"><head><title>目錄</title></head><body><nav>'
            f'<ol><li><a href="ch1.xhtml">目錄條目甲</a></li>'
            f'<li><a href="ch1.xhtml">目錄條目乙</a></li></ol></nav></body></html>'
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": opf,
            "OEBPS/nav.xhtml": nav,
            "OEBPS/ch1.xhtml": _xhtml(h1="第一章", body="真正內文"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "真正內文" in result
        assert "目錄條目甲" not in result  # nav TOC links must not leak in
        assert result.count("## ") == 1   # only the real chapter

    def test_href_with_fragment_and_encoding(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf(
                [("c1", "ch%201.xhtml#start")],  # percent-encoded + fragment
                [("c1", True)],
            ),
            "OEBPS/ch 1.xhtml": _xhtml(h1="章", body="編碼路徑內容"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "編碼路徑內容" in result


# ── TestTextExtraction ──────────────────────────────────────────────────────────

class TestTextExtraction:
    def test_markup_stripped_to_clean_text(self, tmp_path):
        xhtml = (
            f'<html xmlns="{XHTML_NS}"><head><title>t</title>'
            f"<style>.x{{color:red}}</style></head>"
            f"<body><script>var x=1</script>"
            f"<h1>標題</h1><p>第一段<b>粗體</b>文字</p><p>第二段</p></body></html>"
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": xhtml,
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "第一段" in result and "粗體" in result and "第二段" in result
        assert "var x=1" not in result  # script dropped
        assert "color:red" not in result  # style dropped
        assert "<" not in result  # no raw tags

    def test_nested_blocks_not_duplicated(self, tmp_path):
        """blockquote>p / li>p must not double-count text (leaf-block extraction)."""
        chapter = (
            f'<html xmlns="{XHTML_NS}"><head><title>t</title></head><body>'
            f"<h1>章</h1><blockquote><p>引用經文段落</p></blockquote>"
            f"<ul><li><p>清單項目</p></li></ul></body></html>"
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": chapter,
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert result.count("引用經文段落") == 1
        assert result.count("清單項目") == 1

    def test_div_wrapped_body_extracted(self, tmp_path):
        """Prose wrapped only in <div> must not be lost when a heading exists."""
        chapter = (
            f'<html xmlns="{XHTML_NS}"><head><title>t</title></head><body>'
            f"<h1>章標</h1><div>純 div 內文不應遺失</div></body></html>"
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": chapter,
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "純 div 內文不應遺失" in result

    def test_body_fallback_excludes_title(self, tmp_path):
        """Bare-text body fallback must not glue in the <title>."""
        chapter = (
            f'<html xmlns="{XHTML_NS}"><head><title>標題T</title></head>'
            f"<body>裸文字內容</body></html>"
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": chapter,
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "裸文字內容" in result
        assert "標題T裸文字" not in result


# ── TestHeadings ────────────────────────────────────────────────────────────────

class TestHeadings:
    def test_heading_from_h1(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": _xhtml(title="檔名標題", h1="H1 章標", body="x"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "## H1 章標" in result

    def test_heading_from_h3_over_booktitle(self, tmp_path):
        """When chapters title with h3 and <title> repeats the book name, use the
        h3 (real chapter title), not the book-name <title>."""
        chapter = (
            f'<html xmlns="{XHTML_NS}"><head><title>每日得力一分鐘</title></head>'
            f"<body><h3>一月一日</h3><p>當日經文與默想</p></body></html>"
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": chapter,
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "## 一月一日" in result
        assert "## 每日得力一分鐘" not in result

    def test_heading_falls_back_to_title(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": _xhtml(title="TITLE 標題", h1="", body="x"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "## TITLE 標題" in result

    def test_heading_falls_back_to_chapter_number(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": _xhtml(title="", h1="", body="無標題內容"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "## 章節 1" in result


# ── TestMetadata ────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_title_and_author_head(self, tmp_path):
        result = epub.parse(_standard(tmp_path, title="我的書", creator="張三"))
        assert result.startswith("# 我的書")
        assert "作者：張三" in result

    def test_missing_title_omits_head(self, tmp_path):
        result = epub.parse(_standard(tmp_path, title=None, creator=None))
        # no book-title head ("# ..."), but chapter sections ("## ...") remain
        assert not result.startswith("# ")
        assert "作者：" not in result
        assert result.lstrip().startswith("## ")
        assert "第一章" in result


# ── TestSafety ──────────────────────────────────────────────────────────────────

class TestSafety:
    def test_oversize_rejected(self, tmp_path, caplog):
        path = _standard(tmp_path)
        with patch.object(epub, "MAX_DECOMPRESSED_SIZE", 10):
            with caplog.at_level(logging.WARNING):
                result = epub.parse(path)
        assert result == ""
        assert "exceeds" in caplog.text

    def test_not_a_zip(self, tmp_path, caplog):
        p = tmp_path / "bad.epub"
        p.write_bytes(b"this is not a zip")
        with caplog.at_level(logging.WARNING):
            result = epub.parse(str(p))
        assert result == ""
        assert "bad.epub" in caplog.text

    def test_missing_container(self, tmp_path, caplog):
        entries = {"OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)])}
        with caplog.at_level(logging.WARNING):
            result = epub.parse(_build(tmp_path, entries=entries))
        assert result == ""

    def test_xxe_entity_does_not_leak_or_crash(self, tmp_path):
        """A crafted XXE-style external entity in the OPF must not read local
        files or crash — lxml's default (no DTD load) makes the entity undefined,
        and the parser degrades to ''."""
        opf = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE p [<!ENTITY x SYSTEM "file:///etc/hostname">]>'
            f'<package xmlns="{OPF_NS}"><metadata xmlns:dc="{DC_NS}">'
            "<dc:title>&x;</dc:title></metadata><manifest/><spine/></package>"
        )
        entries = {
            "META-INF/container.xml": _container(),
            "OEBPS/content.opf": opf,
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert result == ""  # no local-file content injected, no crash


# ── TestDRM ─────────────────────────────────────────────────────────────────────

class TestDRM:
    def _encryption(self, uri):
        return (
            f'<?xml version="1.0"?>'
            f'<encryption xmlns="{CONTAINER_NS}" xmlns:enc="{ENC_NS}">'
            f'<enc:EncryptedData><enc:CipherData>'
            f'<enc:CipherReference URI="{uri}"/>'
            f"</enc:CipherData></enc:EncryptedData></encryption>"
        )

    def test_content_encrypted_rejected(self, tmp_path, caplog):
        entries = {
            "META-INF/container.xml": _container(),
            "META-INF/encryption.xml": self._encryption("OEBPS/ch1.xhtml"),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": _xhtml(h1="x", body="加密內容"),
        }
        with caplog.at_level(logging.WARNING):
            result = epub.parse(_build(tmp_path, entries=entries))
        assert result == ""
        assert "DRM" in caplog.text

    def test_font_obfuscation_still_extracts(self, tmp_path):
        entries = {
            "META-INF/container.xml": _container(),
            "META-INF/encryption.xml": self._encryption("OEBPS/fonts/font.otf"),
            "OEBPS/content.opf": _opf([("c1", "ch1.xhtml")], [("c1", True)]),
            "OEBPS/ch1.xhtml": _xhtml(h1="第一章", body="字型混淆但內文明文"),
        }
        result = epub.parse(_build(tmp_path, entries=entries))
        assert "字型混淆但內文明文" in result


# ── TestRealEpub: end-to-end against a real .epub (skipped if absent) ───────────

REAL_EPUB = "/Users/jacobmei/Desktop/編年式讀經 靈修 part3.epub"
REAL_EPUB2 = "/Users/jacobmei/Desktop/每日得力一分鐘0503.epub"


@pytest.mark.skipif(not os.path.exists(REAL_EPUB), reason="real .epub fixture absent")
class TestRealEpub:
    def test_extracts_book_metadata_and_ordered_chapters(self):
        result = epub.parse(REAL_EPUB)
        assert result.startswith("# 編年式讀經")  # dc:title head
        assert "作者：" in result                  # dc:creator head
        assert result.count("## ") >= 10           # multiple chapters
        assert "前言" in result                     # real chapter content
        assert "￼" not in result               # image placeholders stripped

    @pytest.mark.skipif(not os.path.exists(REAL_EPUB2), reason="real .epub fixture absent")
    def test_no_adjacent_duplicate_paragraphs(self):
        """Nested-block double-counting (the G3 finding) must not recur on a real book."""
        result = epub.parse(REAL_EPUB2)
        paras = [p for p in result.split("\n\n") if p.strip()]
        dups = [a for a, b in zip(paras, paras[1:])
                if a == b and not a.startswith("##")]
        assert not dups, f"{len(dups)} adjacent duplicate paragraphs"
