"""Security tests for output/epub.py (integrate-epub-output task 3.2).

Pins the "Valid and safe EPUB packaging" requirement: XML escaping on every
dynamic interpolation, fixed zip member order (mimetype first, STORED),
and no exceptions on pathological input.
"""
import io
import xml.etree.ElementTree as ET
import zipfile

from output.epub import _xml_escape, create_epub_archive, render_raw_epub


def _read_member(epub_bytes, name):
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
        return zf.read(name).decode("utf-8")


class TestXmlEscape:
    def test_markup_escaped(self):
        assert _xml_escape('<script>alert("x")</script>') == \
            "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"

    def test_cdata_close_escaped(self):
        assert _xml_escape("]]>") == "]]&gt;"

    def test_ampersand_first(self):
        assert _xml_escape("&lt;") == "&amp;lt;"

    def test_nul_and_control_chars_dropped(self):
        assert _xml_escape("a\x00b\x01c\x0bd") == "abcd"
        assert _xml_escape("keep\ttab\nnewline") == "keep\ttab\nnewline"

    def test_lone_surrogate_dropped(self):
        s = "ok" + "\ud800" + "ok"
        out = _xml_escape(s)
        assert out == "okok"
        out.encode("utf-8")  # must be encodable

    def test_non_str_coerced(self):
        assert _xml_escape(123) == "123"
        assert _xml_escape(None) == "None"


class TestArchiveStructure:
    def test_mimetype_first_and_stored(self):
        data = create_epub_archive("書名", "<p>內容</p>")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            infos = zf.infolist()
            assert infos[0].filename == "mimetype"
            assert infos[0].compress_type == zipfile.ZIP_STORED
            names = {i.filename for i in infos}
            assert {"META-INF/container.xml", "OEBPS/content.opf",
                    "OEBPS/toc.ncx", "OEBPS/nav.xhtml",
                    "OEBPS/text/content.xhtml", "OEBPS/styles.css"} <= names

    def test_metadata_injection_escaped_everywhere(self):
        evil = '</dc:title><script>x</script>"]]>'
        data = create_epub_archive(
            title=evil, content_html="<p>ok</p>",
            summary=evil, tags=[evil], source_path=evil, language=evil)
        for member in ("OEBPS/content.opf", "OEBPS/toc.ncx",
                       "OEBPS/nav.xhtml", "OEBPS/text/content.xhtml"):
            text = _read_member(data, member)
            assert "<script>" not in text
            # every member must stay well-formed XML
            ET.fromstring(text)

    def test_body_content_via_renderer_escaped(self):
        data = render_raw_epub("# 標題\n\n<script>alert(1)</script> & ]]>", "檔名.pdf")
        body = _read_member(data, "OEBPS/text/content.xhtml")
        assert "<script>" not in body
        ET.fromstring(body)


class TestPathologicalInput:
    def test_nul_lone_surrogate_long_line_do_not_raise(self):
        nasty = "行一\x00\ud800" + ("長" * 100000) + "\n\n| a\x01 | b |\n|---|---|\n| \x0b | x |"
        data = render_raw_epub(nasty, "壞\x00檔\ud800.pdf")
        assert zipfile.is_zipfile(io.BytesIO(data))
        body = _read_member(data, "OEBPS/text/content.xhtml")
        ET.fromstring(body)

    def test_non_str_title_and_tags(self):
        data = create_epub_archive(12345, "<p>x</p>", tags=[1, None, "ok"])
        assert zipfile.is_zipfile(io.BytesIO(data))
