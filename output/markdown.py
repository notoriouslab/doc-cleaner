"""
Markdown output renderer — generates structured Markdown with optional YAML frontmatter.

Two modes:
1. AI mode: renders AI-structured output (title, summary, tags, refined content)
2. Raw mode (--ai none): renders extracted text as-is with minimal formatting
"""
import datetime
import logging

logger = logging.getLogger(__name__)


def render_ai_output(data, filename, source_path=None, frontmatter=True):
    """
    Render AI-structured data into Markdown with optional YAML frontmatter.

    Args:
        data: dict with keys: title, summary, refined_markdown, tags (optional)
        filename: original filename (fallback title)
        source_path: original file path (for frontmatter)
        frontmatter: whether to include YAML frontmatter (default: True)

    Returns:
        complete Markdown string
    """
    title = data.get("title") or filename
    summary = data.get("summary") or ""
    refined_markdown = data.get("refined_markdown") or ""
    tags = data.get("tags") or []

    parts = []

    if frontmatter:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        pub_date = now.strftime("%Y-%m-%d %H:%M+08:00")

        tags_yaml = "\n".join(f'  - "{t.replace(chr(34), chr(92)+chr(34))}"' for t in tags) if tags else ""
        tags_block = f"tags:\n{tags_yaml}" if tags_yaml else "tags: []"

        fm = (
            f"---\n"
            f'title: "{title}"\n'
            f'description: "{summary}"\n'
            f'pubDate: "{pub_date}"\n'
            f"draft: true\n"
            f"{tags_block}\n"
            f'source_path: "{source_path or ""}"\n'
            f"---\n"
        )
        parts.append(fm)

    parts.append(f"# {title}\n")

    if summary:
        parts.append(f"> {summary}\n")

    if refined_markdown:
        parts.append(refined_markdown)

    return "\n".join(parts) + "\n"


def render_raw_output(text, filename, source_path=None, frontmatter=True):
    """
    Render raw extracted text into Markdown (no AI processing).

    Used when --ai none is specified. Outputs the text as-is with
    a simple header and optional frontmatter.
    """
    parts = []

    if frontmatter:
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
        pub_date = now.strftime("%Y-%m-%d %H:%M+08:00")

        fm = (
            f"---\n"
            f'title: "{filename}"\n'
            f'description: "Raw extraction"\n'
            f'pubDate: "{pub_date}"\n'
            f"draft: true\n"
            f"tags: []\n"
            f'source_path: "{source_path or ""}"\n'
            f"---\n"
        )
        parts.append(fm)

    parts.append(f"# {filename}\n")
    parts.append(text)

    return "\n".join(parts) + "\n"
