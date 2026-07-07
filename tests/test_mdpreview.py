"""Unit tests for macapp/mdpreview.py — Markdown → safe HTML renderer (D2–D4)."""

from macapp import mdpreview


def r(md):
    return mdpreview.render(md)


class TestBlocks:
    def test_heading_levels(self):
        for level in range(1, 7):
            out = r("#" * level + " Title")
            assert f"<h{level}>Title</h{level}>" in out

    def test_paragraph(self):
        out = r("hello world\nsecond line")
        assert "<p>hello world second line</p>" in out

    def test_paragraphs_split_on_blank(self):
        out = r("para one\n\npara two")
        assert "<p>para one</p>" in out and "<p>para two</p>" in out

    def test_pipe_table_real_delimiter(self):
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        out = r(md)
        assert "<table>" in out
        assert "<th>A</th>" in out and "<th>B</th>" in out
        assert "<td>1</td>" in out and "<td>2</td>" in out

    def test_table_empty_header_row(self):
        md = "| | |\n| --- | --- |\n| x | y |"
        out = r(md)
        assert "<table>" in out
        assert "<th></th>" in out  # empty cell, table not broken
        assert "<td>x</td>" in out

    def test_escaped_pipe_in_cell_kept_as_one_cell(self):
        # parsers escape '|' inside cells as '\\|' — preview must not split on it
        md = "| A | B |\n| --- | --- |\n| a\\|b | c |"
        out = r(md)
        assert "<td>a|b</td>" in out
        assert "<td>c</td>" in out
        assert "<td>a\\</td>" not in out

    def test_escaped_backslash_then_delimiter(self):
        # cell ending in a literal backslash: rendered as '\\\\' + real delimiter
        md = "| P | V |\n| --- | --- |\n| C:\\\\ | x |"
        out = r(md)
        assert "<td>C:\\</td>" in out
        assert "<td>x</td>" in out

    def test_unordered_list(self):
        out = r("- one\n- two")
        assert "<ul>" in out and "<li>one</li>" in out and "<li>two</li>" in out

    def test_ordered_list(self):
        out = r("1. one\n2. two")
        assert "<ol>" in out and "<li>one</li>" in out and "<li>two</li>" in out

    def test_blockquote(self):
        out = r("> quoted text")
        assert "<blockquote>quoted text</blockquote>" in out

    def test_fenced_code(self):
        out = r("```\ncode line\n```")
        assert "<pre><code>code line</code></pre>" in out

    def test_horizontal_rule(self):
        out = r("a\n\n---\n\nb")
        assert "<hr>" in out


class TestInline:
    def test_bold(self):
        assert "<strong>x</strong>" in r("**x**")
        assert "<strong>y</strong>" in r("__y__")

    def test_italic(self):
        assert "<em>x</em>" in r("*x*")

    def test_inline_code(self):
        assert "<code>foo</code>" in r("`foo`")

    def test_code_protects_emphasis(self):
        # asterisks inside code must NOT become <em>
        out = r("`a*b*c`")
        assert "<code>a*b*c</code>" in out
        assert "<em>" not in out

    def test_valid_link(self):
        out = r("[site](https://e.com)")
        assert '<a href="https://e.com">site</a>' in out

    def test_mailto_link(self):
        out = r("[mail](mailto:a@b.com)")
        assert '<a href="mailto:a@b.com">mail</a>' in out


class TestSecurity:
    def test_script_tag_escaped(self):
        out = r("hello <script>alert(1)</script> world")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_script_in_table_cell_escaped(self):
        md = "| A | B |\n| --- | --- |\n| <script>x</script> | ok |"
        out = r(md)
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_script_in_code_block_escaped(self):
        out = r("```\n<script>x</script>\n```")
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    def test_javascript_link_neutralized(self):
        out = r("[click](javascript:alert(1))")
        # No anchor element at all; the text may appear inert but is not a link.
        assert "<a" not in out
        assert "href" not in out

    def test_href_quote_breakout_neutralized(self):
        # scheme is valid (https) but the URL contains a quote → suspicious,
        # so no anchor is emitted, and any quote present is escaped.
        out = r('[x](https://e.com" onmouseover="alert(1))')
        assert "<a" not in out                 # no link emitted for a dirty URL
        assert 'onmouseover="' not in out      # no real (raw-quote) attribute
        assert "&quot;" in out                 # the quote is escaped, not raw

    def test_data_uri_link_neutralized(self):
        out = r("[x](data:text/html,<script>1</script>)")
        assert "<a" not in out


class TestFrontmatter:
    SAMPLE = (
        '---\n'
        'title: "My Doc"\n'
        'description: "Raw extraction"\n'
        'pubDate: "2026-06-12 00:00+08:00"\n'
        'draft: true\n'
        'tags: []\n'
        'sourcePath: "/x/y.txt"\n'
        '---\n'
        '# My Doc\n\nBody text here.'
    )

    def test_title_shown_frontmatter_hidden(self):
        out = r(self.SAMPLE)
        assert "My Doc" in out
        assert "pubDate:" not in out  # raw frontmatter line hidden
        assert "sourcePath:" not in out
        assert "Body text here." in out

    def test_no_frontmatter_renders_from_top(self):
        out = r("# Heading\n\ntext")
        assert "<h1>Heading</h1>" in out

    def test_unterminated_frontmatter_treated_as_body(self):
        # opening --- but no closing --- → do not hang, render as body
        out = r("---\ntitle: x\nstill going")
        assert out != ""  # produced something, no exception/hang


class TestEdge:
    def test_empty_string(self):
        assert r("") == ""

    def test_whitespace_only(self):
        assert r("   \n  \n").strip() == ""

    def test_nul_byte_does_not_crash(self):
        # \x00 is used internally as a code-span sentinel; a literal \x00 in the
        # source (incl. a fake "\x00<digit>\x00" placeholder) must not IndexError.
        assert isinstance(r("a \x000\x00 b"), str)
        assert isinstance(r("real `code` and \x005\x00 fake"), str)

    def test_pathological_unclosed_brackets_is_fast(self):
        # ReDoS guard: a huge single run of "[a](" must not hang (O(n^2) link
        # regex backtracking) — the oversized run is escaped, not parsed.
        out = r("[a](" * 60000)  # ~240KB single line, within byte cap
        assert isinstance(out, str) and len(out) > 0
