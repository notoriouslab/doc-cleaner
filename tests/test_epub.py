import unittest
import zipfile
import io
import xml.etree.ElementTree as ET
from output.epub import fallback_markdown_to_xhtml, create_epub_archive, render_ai_epub, render_raw_epub


class TestEPUBGeneration(unittest.TestCase):

    def test_fallback_markdown_to_xhtml_basic(self):
        markdown_text = (
            "# Heading 1\n"
            "## Heading 2\n"
            "This is a paragraph with **bold** and *italic* text and `code`.\n"
            "\n"
            "> This is a blockquote\n"
            "\n"
            "- List item 1\n"
            "- List item 2\n"
        )
        xhtml = fallback_markdown_to_xhtml(markdown_text)
        
        self.assertIn("<h1>Heading 1</h1>", xhtml)
        self.assertIn("<h2>Heading 2</h2>", xhtml)
        self.assertIn("<strong>bold</strong>", xhtml)
        self.assertIn("<em>italic</em>", xhtml)
        self.assertIn("<code>code</code>", xhtml)
        self.assertIn("<blockquote>This is a blockquote</blockquote>", xhtml)
        self.assertIn("<ul>", xhtml)
        self.assertIn("<li>List item 1</li>", xhtml)

    def test_fallback_markdown_to_xhtml_table(self):
        markdown_text = (
            "| Header 1 | Header 2 |\n"
            "|---|---|\n"
            "| Cell 1 | Cell 2 |\n"
        )
        xhtml = fallback_markdown_to_xhtml(markdown_text)
        
        self.assertIn("<table>", xhtml)
        self.assertIn("<thead><tr>", xhtml)
        self.assertIn("<th>Header 1</th>", xhtml)
        self.assertIn("<tbody>", xhtml)
        self.assertIn("<td>Cell 1</td>", xhtml)
        self.assertIn("</table>", xhtml)

    def test_fallback_markdown_to_xhtml_code_block(self):
        markdown_text = (
            "```python\n"
            "def hello():\n"
            "    print('world')\n"
            "```\n"
        )
        xhtml = fallback_markdown_to_xhtml(markdown_text)
        self.assertIn("<pre><code>", xhtml)
        self.assertIn("def hello():", xhtml)
        self.assertIn("</code></pre>", xhtml)

    def test_create_epub_archive(self):
        title = "Test Book"
        content_html = "<p>Hello World</p>"
        summary = "A test book summary"
        tags = ["test", "epub"]
        source_path = "test_doc.pdf"
        
        epub_bytes = create_epub_archive(
            title=title,
            content_html=content_html,
            summary=summary,
            tags=tags,
            source_path=source_path,
            language="zh-TW"
        )
        
        self.assertIsInstance(epub_bytes, bytes)
        
        # Verify zip structure
        zip_buf = io.BytesIO(epub_bytes)
        with zipfile.ZipFile(zip_buf, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("mimetype", namelist)
            self.assertIn("META-INF/container.xml", namelist)
            self.assertIn("OEBPS/content.opf", namelist)
            self.assertIn("OEBPS/toc.ncx", namelist)
            self.assertIn("OEBPS/nav.xhtml", namelist)
            self.assertIn("OEBPS/styles.css", namelist)
            self.assertIn("OEBPS/text/content.xhtml", namelist)
            
            # Verify mimetype content is not compressed and contains application/epub+zip
            info = zf.getinfo("mimetype")
            self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
            mimetype_content = zf.read("mimetype").decode("utf-8")
            self.assertEqual(mimetype_content, "application/epub+zip")
            
            # Verify container.xml is valid XML
            container_content = zf.read("META-INF/container.xml").decode("utf-8")
            ET.fromstring(container_content)
            
            # Verify content.opf is valid XML and contains metadata
            opf_content = zf.read("OEBPS/content.opf").decode("utf-8")
            self.assertIn(title, opf_content)
            self.assertIn("zh-TW", opf_content)
            ET.fromstring(opf_content)
            
            # Verify content page lists metadata and content
            content_xhtml = zf.read("OEBPS/text/content.xhtml").decode("utf-8")
            self.assertIn("Test Book", content_xhtml)
            self.assertIn("A test book summary", content_xhtml)
            self.assertIn('<span class="tag">test</span>', content_xhtml)
            self.assertIn("<p>Hello World</p>", content_xhtml)

    def test_render_ai_epub(self):
        data = {
            "title": "AI Document",
            "summary": "AI generated summary",
            "refined_markdown": "# AI Section\nThis is content.",
            "tags": ["ai", "clean"]
        }
        epub_bytes = render_ai_epub(data, filename="original.pdf")
        self.assertIsInstance(epub_bytes, bytes)

    def test_render_raw_epub(self):
        text = "This is raw text content."
        epub_bytes = render_raw_epub(text, filename="raw.txt")
        self.assertIsInstance(epub_bytes, bytes)


if __name__ == "__main__":
    unittest.main()
