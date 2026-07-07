"""
EPUB output renderer — generates structured EPUB 3 files.

Uses Python's standard `zipfile` library to assemble the EPUB, and a single
built-in Markdown→XHTML renderer (zero dependencies) targeting the constructs
doc-cleaner itself generates: headings, pipe tables, lists, and code blocks.
One rendering path means identical output in CLI, development, and packaged
apps.
"""
import datetime
import uuid
import re
import zipfile
import io
import logging

logger = logging.getLogger(__name__)

# Characters forbidden in XML 1.0 element content: C0 controls (except \t\n\r),
# lone surrogates (reachable via surrogateescape-decoded input; they also make
# UTF-8 encoding raise), and the two noncharacters.
_XML_INVALID = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\ud800-\udfff\ufffe\uffff]")


def _xml_escape(value):
    """Coerce to str, drop XML-invalid characters, escape markup. Never raises."""
    s = _XML_INVALID.sub("", str(value))
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def fallback_markdown_to_xhtml(text):
    """
    A lightweight, regex-based Markdown to XHTML converter.
    Guarantees valid XML output and zero external dependencies.
    """
    xml_escape = _xml_escape

    def inline_format(s):
        # Escaping code snippets separately first to avoid formatting conflicts
        # Inline code `code`
        s = re.sub(r"`([^`]+)`", lambda m: f"<code>{xml_escape(m.group(1))}</code>", s)
        # Bold **
        s = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", s)
        # Italic * or _
        s = re.sub(r"\*(.*?)\*", r"<em>\1</em>", s)
        s = re.sub(r"_(.*?)_", r"<em>\1</em>", s)
        return s

    lines = text.splitlines()
    blocks = []
    in_code_block = False
    in_list = None  # 'ul', 'ol', or None
    in_table = False
    table_rows = []

    def flush_list():
        nonlocal in_list
        if in_list:
            blocks.append(f"</{in_list}>")
            in_list = None

    def flush_table():
        nonlocal in_table, table_rows
        if in_table:
            if len(table_rows) > 0:
                html = ["<table>"]
                has_header = False
                if len(table_rows) > 1:
                    second_row = table_rows[1].strip()
                    if re.match(r"^\|?\s*(:?-+:?\s*\|?\s*)+$", second_row):
                        has_header = True
                
                start_idx = 0
                if has_header:
                    header_cells = [c.strip() for c in table_rows[0].split("|")]
                    if header_cells and header_cells[0] == "":
                        header_cells = header_cells[1:]
                    if header_cells and header_cells[-1] == "":
                        header_cells = header_cells[:-1]
                    
                    html.append("<thead><tr>")
                    for cell in header_cells:
                        html.append(f"<th>{inline_format(xml_escape(cell))}</th>")
                    html.append("</tr></thead>")
                    start_idx = 2  # skip header and separator
                
                html.append("<tbody>")
                for row_idx in range(start_idx, len(table_rows)):
                    row = table_rows[row_idx]
                    cells = [c.strip() for c in row.split("|")]
                    if cells and cells[0] == "":
                        cells = cells[1:]
                    if cells and cells[-1] == "":
                        cells = cells[:-1]
                    
                    html.append("<tr>")
                    for cell in cells:
                        html.append(f"<td>{inline_format(xml_escape(cell))}</td>")
                    html.append("</tr>")
                html.append("</tbody>")
                html.append("</table>")
                blocks.append("\n".join(html))
            
            table_rows = []
            in_table = False

    for line in lines:
        stripped = line.strip()

        # Code blocks
        if stripped.startswith("```"):
            flush_list()
            flush_table()
            if in_code_block:
                blocks.append("</code></pre>")
                in_code_block = False
            else:
                blocks.append("<pre><code>")
                in_code_block = True
            continue

        if in_code_block:
            blocks.append(xml_escape(line))
            continue

        # Tables
        if stripped.startswith("|") and not in_code_block:
            flush_list()
            in_table = True
            table_rows.append(line)
            continue
        elif in_table:
            flush_table()

        # Headers
        header_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if header_match:
            flush_list()
            level = len(header_match.group(1))
            content = inline_format(xml_escape(header_match.group(2)))
            blocks.append(f"<h{level}>{content}</h{level}>")
            continue

        # Blockquotes
        if stripped.startswith(">"):
            flush_list()
            content = inline_format(xml_escape(stripped[1:].strip()))
            blocks.append(f"<blockquote>{content}</blockquote>")
            continue

        # Bullet lists
        list_match = re.match(r"^([*\-+])\s+(.*)$", stripped)
        if list_match:
            if in_list != "ul":
                flush_list()
                blocks.append("<ul>")
                in_list = "ul"
            content = inline_format(xml_escape(list_match.group(2)))
            blocks.append(f"<li>{content}</li>")
            continue

        # Ordered lists
        ol_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ol_match:
            if in_list != "ol":
                flush_list()
                blocks.append("<ol>")
                in_list = "ol"
            content = inline_format(xml_escape(ol_match.group(2)))
            blocks.append(f"<li>{content}</li>")
            continue

        # Empty lines
        if not stripped:
            flush_list()
            continue

        # Regular paragraphs
        flush_list()
        content = inline_format(xml_escape(line))
        blocks.append(f"<p>{content}</p>")

    flush_list()
    flush_table()

    return "\n".join(blocks)


def markdown_to_xhtml(text):
    """Convert markdown text to valid XHTML string (single built-in renderer)."""
    return fallback_markdown_to_xhtml(text)


def create_epub_archive(title, content_html, summary="", tags=None, source_path="", language="zh-TW"):
    """
    Generate the EPUB binary data as a zip archive in memory.
    """
    epub_buf = io.BytesIO()
    
    with zipfile.ZipFile(epub_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. mimetype (MUST be the first file and stored uncompressed)
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        
        # 2. META-INF/container.xml
        container_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
            '  <rootfiles>\n'
            '    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>\n'
            '  </rootfiles>\n'
            '</container>'
        )
        zf.writestr("META-INF/container.xml", container_xml)
        
        # Metadata fields
        book_uuid = str(uuid.uuid4())
        modified_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        xml_esc = _xml_escape

        esc_title = xml_esc(title)
        
        # 3. OEBPS/content.opf
        content_opf = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="pub-id" version="3.0">\n'
            '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
            f'    <dc:identifier id="pub-id">urn:uuid:{book_uuid}</dc:identifier>\n'
            f'    <dc:title>{esc_title}</dc:title>\n'
            f'    <dc:language>{xml_esc(language)}</dc:language>\n'
            '    <dc:creator id="creator">doc-cleaner</dc:creator>\n'
            f'    <meta property="dcterms:modified">{modified_time}</meta>\n'
            '  </metadata>\n'
            '  <manifest>\n'
            '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>\n'
            '    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>\n'
            '    <item id="content" href="text/content.xhtml" media-type="application/xhtml+xml"/>\n'
            '    <item id="css" href="styles.css" media-type="text/css"/>\n'
            '  </manifest>\n'
            '  <spine toc="ncx">\n'
            '    <itemref idref="nav"/>\n'
            '    <itemref idref="content"/>\n'
            '  </spine>\n'
            '</package>'
        )
        zf.writestr("OEBPS/content.opf", content_opf)
        
        # 4. OEBPS/toc.ncx
        toc_ncx = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
            '  <head>\n'
            f'    <meta name="dtb:uid" content="urn:uuid:{book_uuid}"/>\n'
            '    <meta name="dtb:depth" content="1"/>\n'
            '    <meta name="dtb:totalPageCount" content="0"/>\n'
            '    <meta name="dtb:maxPageNumber" content="0"/>\n'
            '  </head>\n'
            '  <docTitle>\n'
            f'    <text>{esc_title}</text>\n'
            '  </docTitle>\n'
            '  <navMap>\n'
            '    <navPoint id="navpoint-1" playOrder="1">\n'
            '      <navLabel>\n'
            '        <text>Table of Contents</text>\n'
            '      </navLabel>\n'
            '      <content src="nav.xhtml"/>\n'
            '    </navPoint>\n'
            '    <navPoint id="navpoint-2" playOrder="2">\n'
            '      <navLabel>\n'
            f'        <text>{esc_title}</text>\n'
            '      </navLabel>\n'
            '      <content src="text/content.xhtml"/>\n'
            '    </navPoint>\n'
            '  </navMap>\n'
            '</ncx>'
        )
        zf.writestr("OEBPS/toc.ncx", toc_ncx)
        
        # 5. OEBPS/nav.xhtml
        nav_xhtml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE html>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{xml_esc(language)}" xml:lang="{xml_esc(language)}">\n'
            '<head>\n'
            f'  <title>{esc_title} - Table of Contents</title>\n'
            '  <meta charset="utf-8" />\n'
            '  <link rel="stylesheet" href="styles.css" type="text/css" />\n'
            '</head>\n'
            '<body>\n'
            '  <nav epub:type="toc" id="toc">\n'
            '    <h1>Table of Contents</h1>\n'
            '    <ol>\n'
            f'      <li><a href="text/content.xhtml">{esc_title}</a></li>\n'
            '    </ol>\n'
            '  </nav>\n'
            '</body>\n'
            '</html>'
        )
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml)
        
        # 6. OEBPS/styles.css
        styles_css = (
            'body {\n'
            '  font-family: "Outfit", "Inter", "Helvetica Neue", Helvetica, Arial, "PingFang TC", "Microsoft JhengHei", sans-serif;\n'
            '  line-height: 1.6;\n'
            '  color: #333333;\n'
            '  margin: 0;\n'
            '  padding: 1.5em;\n'
            '}\n'
            'h1, h2, h3, h4, h5, h6 {\n'
            '  color: #111111;\n'
            '  font-weight: 700;\n'
            '  margin-top: 1.5em;\n'
            '  margin-bottom: 0.5em;\n'
            '}\n'
            'h1 {\n'
            '  font-size: 1.8em;\n'
            '  border-bottom: 1px solid #eaeaea;\n'
            '  padding-bottom: 0.3em;\n'
            '}\n'
            'h2 {\n'
            '  font-size: 1.4em;\n'
            '  border-bottom: 1px solid #f0f0f0;\n'
            '  padding-bottom: 0.2em;\n'
            '}\n'
            'h3 {\n'
            '  font-size: 1.2em;\n'
            '}\n'
            'p {\n'
            '  margin-top: 0;\n'
            '  margin-bottom: 1em;\n'
            '  text-align: justify;\n'
            '}\n'
            'blockquote {\n'
            '  margin: 1.5em 0;\n'
            '  padding: 0 1em;\n'
            '  color: #666666;\n'
            '  border-left: 4px solid #dddddd;\n'
            '  font-style: italic;\n'
            '}\n'
            'code {\n'
            '  font-family: Consolas, Monaco, "Andale Mono", monospace;\n'
            '  background-color: #f6f8fa;\n'
            '  padding: 0.2em 0.4em;\n'
            '  border-radius: 3px;\n'
            '  font-size: 0.9em;\n'
            '}\n'
            'pre {\n'
            '  background-color: #f6f8fa;\n'
            '  padding: 1em;\n'
            '  border-radius: 6px;\n'
            '  overflow-x: auto;\n'
            '}\n'
            'pre code {\n'
            '  background-color: transparent;\n'
            '  padding: 0;\n'
            '  border-radius: 0;\n'
            '  font-size: 0.85em;\n'
            '}\n'
            'table {\n'
            '  border-collapse: collapse;\n'
            '  width: 100%;\n'
            '  margin: 1.5em 0;\n'
            '  font-size: 0.9em;\n'
            '}\n'
            'th, td {\n'
            '  border: 1px solid #dddddd;\n'
            '  padding: 8px 12px;\n'
            '  text-align: left;\n'
            '}\n'
            'th {\n'
            '  background-color: #f2f2f2;\n'
            '  font-weight: bold;\n'
            '}\n'
            'tr:nth-child(even) {\n'
            '  background-color: #fafafa;\n'
            '}\n'
            '.tags {\n'
            '  margin-top: 1.5em;\n'
            '  font-size: 0.85em;\n'
            '  color: #666666;\n'
            '}\n'
            '.tag {\n'
            '  display: inline-block;\n'
            '  background-color: #e1ecf4;\n'
            '  color: #39739d;\n'
            '  padding: 2px 6px;\n'
            '  border-radius: 3px;\n'
            '  margin-right: 5px;\n'
            '  margin-bottom: 5px;\n'
            '}\n'
            '.metadata-block {\n'
            '  background-color: #fafafa;\n'
            '  border-left: 3px solid #0066cc;\n'
            '  padding: 0.8em 1em;\n'
            '  margin-bottom: 2em;\n'
            '  border-radius: 0 4px 4px 0;\n'
            '}\n'
            '.metadata-title {\n'
            '  font-weight: bold;\n'
            '  font-size: 0.95em;\n'
            '  color: #555555;\n'
            '  margin-bottom: 0.3em;\n'
            '}\n'
            '.metadata-value {\n'
            '  font-size: 0.9em;\n'
            '  color: #666666;\n'
            '  margin-bottom: 0.5em;\n'
            '}\n'
        )
        zf.writestr("OEBPS/styles.css", styles_css)
        
        # 7. OEBPS/text/content.xhtml
        body_parts = [f"<h1>{esc_title}</h1>"]
        
        if summary or source_path or tags:
            body_parts.append('<div class="metadata-block">')
            if source_path:
                body_parts.append(f'  <div class="metadata-title">Source File</div>')
                body_parts.append(f'  <div class="metadata-value">{xml_esc(source_path)}</div>')
            if summary:
                body_parts.append(f'  <div class="metadata-title">Summary</div>')
                body_parts.append(f'  <div class="metadata-value">{xml_esc(summary)}</div>')
            if tags:
                body_parts.append(f'  <div class="metadata-title">Tags</div>')
                tag_spans = "".join(f'<span class="tag">{xml_esc(t)}</span>' for t in tags)
                body_parts.append(f'  <div class="metadata-value">{tag_spans}</div>')
            body_parts.append('</div>')
            
        body_parts.append(content_html)
        
        content_xhtml_body = "\n".join(body_parts)
        
        content_xhtml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<!DOCTYPE html>\n'
            f'<html xmlns="http://www.w3.org/1999/xhtml" lang="{xml_esc(language)}" xml:lang="{xml_esc(language)}">\n'
            '<head>\n'
            f'  <title>{esc_title}</title>\n'
            '  <meta charset="utf-8" />\n'
            '  <link rel="stylesheet" href="../styles.css" type="text/css" />\n'
            '</head>\n'
            '<body>\n'
            f'  <section class="chapter">\n'
            f'    {content_xhtml_body}\n'
            '  </section>\n'
            '</body>\n'
            '</html>'
        )
        zf.writestr("OEBPS/text/content.xhtml", content_xhtml)
        
    return epub_buf.getvalue()


def render_ai_epub(data, filename, source_path=None, language="zh-TW"):
    """Render structured AI JSON response to EPUB byte content."""
    title = data.get("title") or filename
    summary = data.get("summary") or ""
    refined_markdown = data.get("refined_markdown") or ""
    tags = data.get("tags") or []
    
    content_html = markdown_to_xhtml(refined_markdown)
    return create_epub_archive(
        title=title,
        content_html=content_html,
        summary=summary,
        tags=tags,
        source_path=source_path or filename,
        language=language
    )


def render_raw_epub(text, filename, source_path=None, language="zh-TW"):
    """Render raw extracted text to EPUB byte content."""
    content_html = markdown_to_xhtml(text)
    return create_epub_archive(
        title=filename,
        content_html=content_html,
        summary="Raw extraction",
        tags=[],
        source_path=source_path or filename,
        language=language
    )
