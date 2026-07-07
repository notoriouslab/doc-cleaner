"""
Markdown → safe HTML renderer for the in-app preview (D1–D4).

Renders the app's own Markdown output into a whitelisted, HTML-escaped subset
that is safe to inject into the pywebview DOM via innerHTML. This is NOT a
general CommonMark implementation — it covers the constructs the converters
emit (headings, paragraphs, pipe tables, lists, blockquotes, fenced code,
inline emphasis/code/links, horizontal rules). Anything unmatched degrades to
escaped paragraph text — readable, never broken markup.

Security (the trust boundary): every text run is HTML-escaped *before* being
wrapped in a tag, only a fixed tag set is emitted, links are produced only for
http/https/mailto schemes, and the href value itself is escaped so a
scheme-valid URL cannot break out of the attribute. A `.md` containing
`<script>` (anywhere — paragraph, table cell, code block) renders as inert text.

This module MUST NOT import `webview`: it is pure logic, unit-tested.
"""
import html
import re

# Upper bound for the preview bridge: files larger than this are not read.
MAX_PREVIEW_BYTES = 5_000_000

# A single text run longer than this skips inline parsing (just escaped). The
# inline regexes (notably the link pattern) backtrack quadratically on
# pathological unclosed-bracket input, so a multi-hundred-KB line within the
# byte cap could otherwise freeze the UI thread. Real output never approaches it.
_MAX_INLINE = 50_000

_ALLOWED_SCHEMES = ("http://", "https://", "mailto:")

# A table delimiter row as the parsers actually emit it: "| --- | --- |"
# (surrounding spaces, >=1 dash, optional alignment colons) — NOT a strict "|---|".
_TABLE_DELIM_RE = re.compile(r"^\s*\|(\s*:?-+:?\s*\|)+\s*$")
_HEADING_RE = re.compile(r"(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")
_UL_RE = re.compile(r"^\s*[-*+]\s+")
_OL_RE = re.compile(r"^\s*\d+\.\s+")


# ── inline ──────────────────────────────────────────────────────────────────

def _link_sub(m):
    label, url = m.group(1), m.group(2)
    # url here is already HTML-escaped. Defense in depth: a clean URL has no raw
    # quote / angle bracket / whitespace, so reject (render inert) if the escaped
    # url shows any of those entities — never emit a link for a suspicious URL.
    if any(bad in url for bad in ("&quot;", "&lt;", "&gt;", "&#x27;", " ", "\t")):
        return m.group(0)
    # Scheme whitelist; the prefixes have no special chars so escaping is moot.
    if url.lower().startswith(_ALLOWED_SCHEMES):
        return f'<a href="{url}">{label}</a>'  # url already escaped → no breakout
    return m.group(0)  # disallowed scheme → leave inert (already escaped) text


def _inline(text):
    """Render inline markup from a raw text run into safe HTML."""
    # Pathological-input guard: skip inline regexes for oversized runs (the link
    # pattern backtracks quadratically on unclosed brackets) — just escape.
    if len(text) > _MAX_INLINE:
        return html.escape(text, quote=True)
    # 1) Stash code spans (escaped content) so * _ inside them aren't formatted.
    codes = []

    def _stash(m):
        codes.append(html.escape(m.group(1), quote=True))
        return f"\x00{len(codes) - 1}\x00"

    tmp = re.sub(r"`([^`]+)`", _stash, text)
    # 2) Escape everything else (markers like * _ [ ] ( ) survive escaping).
    tmp = html.escape(tmp, quote=True)
    # 3) Links — operate on the escaped string; href value stays escaped.
    tmp = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", _link_sub, tmp)
    # 4) Bold before italic.
    tmp = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", tmp)
    tmp = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", tmp)
    tmp = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", tmp)
    tmp = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", tmp)
    # 5) Restore code spans.
    tmp = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", tmp)
    return tmp


# ── tables ──────────────────────────────────────────────────────────────────

def _split_row(line):
    # Split on unescaped '|' only; backslash escapes the next character
    # ('\|' → literal pipe, '\\' → literal backslash), matching how the
    # PDF/DOCX/PPTX parsers escape pipes inside table cells.
    cells, cur, i = [], [], 0
    s = line.strip()
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            cur.append(s[i + 1])
            i += 2
        elif ch == "|":
            cells.append("".join(cur).strip())
            cur = []
            i += 1
        else:
            cur.append(ch)
            i += 1
    cells.append("".join(cur).strip())
    # Drop the empty edge cells produced by leading/trailing delimiters
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _render_table(table_lines):
    header = _split_row(table_lines[0])
    th = "".join(f"<th>{_inline(c)}</th>" for c in header)
    body = ""
    for line in table_lines[2:]:  # skip header + delimiter
        tds = "".join(f"<td>{_inline(c)}</td>" for c in _split_row(line))
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


# ── frontmatter ───────────────────────────────────────────────────────────────

def _split_frontmatter(text):
    """Return (title_or_None, body). A leading ---...--- block is parsed for
    its title and stripped from the body. Unterminated → no frontmatter."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return None, text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            title = _frontmatter_title(lines[1:i])
            return title, "\n".join(lines[i + 1:])
    return None, text  # no closing fence → treat all as body


def _frontmatter_title(fm_lines):
    for ln in fm_lines:
        m = re.match(r'\s*title:\s*"?(.*?)"?\s*$', ln)
        if m:
            return m.group(1).replace('\\"', '"').replace("\\\\", "\\")
    return None


# ── block walker ──────────────────────────────────────────────────────────────

def _starts_block(line, nextline):
    s = line.strip()
    if not s:
        return True
    if s.startswith("```"):
        return True
    if _HEADING_RE.match(line):
        return True
    if _HR_RE.match(s):
        return True
    if s.startswith(">"):
        return True
    if _UL_RE.match(line) or _OL_RE.match(line):
        return True
    if "|" in line and _TABLE_DELIM_RE.match(nextline):
        return True
    return False


def _render_blocks(text):
    lines = text.split("\n")
    n = len(lines)
    out = []
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Fenced code block (verbatim, escaped).
        if stripped.startswith("```"):
            i += 1
            code = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # consume closing fence (if any)
            out.append(f"<pre><code>{html.escape(chr(10).join(code), quote=True)}</code></pre>")
            continue

        # Heading.
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2).strip())}</h{level}>")
            i += 1
            continue

        # Horizontal rule.
        if _HR_RE.match(stripped):
            out.append("<hr>")
            i += 1
            continue

        # Table (header row + delimiter row).
        if "|" in line and i + 1 < n and _TABLE_DELIM_RE.match(lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < n and lines[i].strip() and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            out.append(_render_table(table_lines))
            continue

        # Blockquote.
        if stripped.startswith(">"):
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(quote).strip())}</blockquote>")
            continue

        # Unordered list.
        if _UL_RE.match(line):
            items = []
            while i < n and _UL_RE.match(lines[i]):
                items.append(_UL_RE.sub("", lines[i]).strip())
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(it)}</li>" for it in items) + "</ul>")
            continue

        # Ordered list.
        if _OL_RE.match(line):
            items = []
            while i < n and _OL_RE.match(lines[i]):
                items.append(_OL_RE.sub("", lines[i]).strip())
                i += 1
            out.append("<ol>" + "".join(f"<li>{_inline(it)}</li>" for it in items) + "</ol>")
            continue

        # Paragraph (collect until a blank line or a new block).
        para = []
        while i < n and lines[i].strip() and not _starts_block(lines[i], lines[i + 1] if i + 1 < n else ""):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
        else:
            i += 1  # defensive: never spin on an unconsumed line
    return out


def render(markdown_text):
    """Render our Markdown output into a safe, whitelisted HTML string."""
    if not markdown_text:
        return ""
    # Strip NUL: it has no Markdown meaning, and the inline code-span stash uses
    # \x00 sentinels — a literal \x00 in the source would otherwise collide.
    markdown_text = markdown_text.replace("\x00", "")
    title, body = _split_frontmatter(markdown_text)
    parts = []
    if title:
        parts.append(f'<p class="meta">{html.escape(title, quote=True)}</p>')
    parts.extend(_render_blocks(body))
    return "\n".join(parts)
