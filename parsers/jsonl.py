"""
Claude Code session transcript parser (.jsonl).

Converts Claude Code JSONL transcript files to structured Markdown.
Handles text, tool_use (summary), thinking (collapsible), and tool_result
(collapsible) blocks.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_MAX_INPUT_LEN = 60


def _extract_timestamp(ts_str):
    """Parse ISO 8601 timestamp to local HH:MM. Returns '' on failure."""
    if not ts_str:
        return ""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        return local_dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return ""


def _truncate(text, max_len=_MAX_INPUT_LEN):
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _tool_summary(block):
    """Return '🔧 ToolName: primary_input' for a tool_use block."""
    name = block.get("name", "Tool")
    inp = block.get("input") or {}

    if not isinstance(inp, dict) or not inp:
        return f"🔧 {name}"

    if "file_path" in inp:
        return f"🔧 {name}: {_truncate(inp['file_path'])}"
    if "command" in inp:
        return f"🔧 {name}: {_truncate(inp['command'])}"
    if "url" in inp:
        return f"🔧 {name}: {_truncate(inp['url'])}"
    if "prompt" in inp:
        return f"🔧 {name}: {_truncate(inp['prompt'])}"

    # fallback: first value that is a string
    for v in inp.values():
        if isinstance(v, str):
            return f"🔧 {name}: {_truncate(v)}"

    return f"🔧 {name}"


def _tool_result_body(block):
    """Extract text from a tool_result block. Returns '' if empty."""
    result_content = block.get("content", "")
    if isinstance(result_content, str):
        return result_content.strip()
    if isinstance(result_content, list):
        parts = [
            rc.get("text", "").strip()
            for rc in result_content
            if rc.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _render_user(msg, ts, tool_name_map=None):
    if tool_name_map is None:
        tool_name_map = {}

    content = msg.get("content", "")
    header = f"**User** · {ts}" if ts else "**User**"

    if isinstance(content, str):
        body = content.strip()
        return f"{header}\n\n{body}" if body else header

    if not isinstance(content, list):
        return header

    parts = [header]
    for c in content:
        ctype = c.get("type")
        if ctype == "text":
            t = c.get("text", "").strip()
            if t:
                parts.append(t)
        elif ctype == "tool_result":
            tool_id = c.get("tool_use_id", "")
            tool_name = tool_name_map.get(tool_id, "tool")
            body = _tool_result_body(c)
            if body:
                parts.append(
                    f"<details><summary>tool_result ({tool_name})</summary>\n\n{body}\n\n</details>"
                )

    return "\n\n".join(parts)


def _render_assistant(msg, ts):
    content = msg.get("content", [])
    if not isinstance(content, list):
        content = []

    header = f"**Assistant** · {ts}" if ts else "**Assistant**"
    parts = [header]

    for block in content:
        btype = block.get("type")
        if btype == "text":
            t = block.get("text", "").strip()
            if t:
                parts.append(t)
        elif btype == "tool_use":
            parts.append(_tool_summary(block))
        elif btype == "thinking":
            t = (block.get("thinking") or "").strip()
            if t:
                parts.append(
                    f"<details><summary>thinking</summary>\n\n{t}\n\n</details>"
                )

    return "\n\n".join(parts)


def _render_session(session_id, messages):
    """Render one session's messages as a Markdown section."""
    short_id = session_id[:8] if session_id else "unknown"

    # Build tool_use_id -> name map for tool_result labelling
    tool_name_map = {}
    for entry in messages:
        if entry.get("type") == "assistant":
            msg_content = entry.get("message", {}).get("content", [])
            if isinstance(msg_content, list):
                for block in msg_content:
                    if block.get("type") == "tool_use":
                        bid = block.get("id", "")
                        if bid:
                            tool_name_map[bid] = block.get("name", "tool")

    # Date from first message timestamp
    first_ts = next((m.get("timestamp", "") for m in messages), "")
    date_str = ""
    if first_ts:
        try:
            dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00")).astimezone()
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    lines = [f"## Session {short_id}"]
    if date_str:
        lines.append(f"_{date_str}_")
    lines.append("")

    rendered = []
    for entry in messages:
        ts = _extract_timestamp(entry.get("timestamp", ""))
        msg = entry.get("message", {})
        role = entry.get("type")

        if role == "user":
            rendered.append(_render_user(msg, ts, tool_name_map))
        elif role == "assistant":
            rendered.append(_render_assistant(msg, ts))

    lines.append("\n\n---\n\n".join(rendered))
    return "\n".join(lines)


def parse(filepath):
    """
    Parse a Claude Code JSONL transcript file into structured Markdown.

    Groups messages by sessionId, renders user/assistant turns with
    tool summaries, collapsible thinking blocks, and collapsible tool results.
    """
    session_order = []
    sessions = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping malformed JSON at line {lineno}")
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            sid = entry.get("sessionId", "")
            if sid not in sessions:
                session_order.append(sid)
            sessions[sid].append(entry)

    if not session_order:
        logger.warning(f"No conversation messages found in {filepath}")
        return ""

    sections = []
    for sid in session_order:
        sections.append(_render_session(sid, sessions[sid]))

    return "\n\n---\n\n".join(sections)
