"""Unit tests for parsers/jsonl.py — Claude Code transcript parser."""
import json
import os
import tempfile

import pytest

from parsers.jsonl import parse, _tool_summary, _extract_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(lines, tmp_path):
    path = os.path.join(tmp_path, "session.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def _make_user(content, session_id="abc123", timestamp="2026-06-05T01:58:19.000Z"):
    return {
        "type": "user",
        "sessionId": session_id,
        "timestamp": timestamp,
        "message": {"role": "user", "content": content},
    }


def _make_assistant(content_blocks, session_id="abc123", timestamp="2026-06-05T01:59:00.000Z"):
    return {
        "type": "assistant",
        "sessionId": session_id,
        "timestamp": timestamp,
        "message": {"role": "assistant", "content": content_blocks},
    }


# ---------------------------------------------------------------------------
# _extract_timestamp
# ---------------------------------------------------------------------------

def test_extract_timestamp_utc():
    result = _extract_timestamp("2026-06-05T01:58:19.169Z")
    assert ":" in result  # HH:MM format


def test_extract_timestamp_missing():
    assert _extract_timestamp("") == ""
    assert _extract_timestamp(None) == ""


def test_extract_timestamp_malformed():
    assert _extract_timestamp("not-a-date") == ""


# ---------------------------------------------------------------------------
# _tool_summary
# ---------------------------------------------------------------------------

def test_tool_summary_file_path():
    block = {"name": "Read", "input": {"file_path": "/foo/bar.py"}}
    assert _tool_summary(block) == "🔧 Read: /foo/bar.py"


def test_tool_summary_command():
    block = {"name": "Bash", "input": {"command": "git status"}}
    assert _tool_summary(block) == "🔧 Bash: git status"


def test_tool_summary_url_priority_over_prompt():
    block = {"name": "WebFetch", "input": {"url": "https://example.com", "prompt": "summarise this page"}}
    assert _tool_summary(block) == "🔧 WebFetch: https://example.com"


def test_tool_summary_prompt_when_no_url():
    block = {"name": "Agent", "input": {"prompt": "run the tests"}}
    assert _tool_summary(block) == "🔧 Agent: run the tests"


def test_tool_summary_fallback_first_string_value():
    block = {"name": "Agent", "input": {"description": "run tests", "other": 42}}
    assert _tool_summary(block) == "🔧 Agent: run tests"


def test_tool_summary_empty_input():
    block = {"name": "Unknown", "input": {}}
    assert _tool_summary(block) == "🔧 Unknown"


def test_tool_summary_none_input():
    block = {"name": "Unknown", "input": None}
    assert _tool_summary(block) == "🔧 Unknown"


def test_tool_summary_truncates_long_value():
    long_cmd = "echo " + "x" * 100
    block = {"name": "Bash", "input": {"command": long_cmd}}
    result = _tool_summary(block)
    assert len(result) < len(long_cmd) + 10
    assert result.endswith("…")


# ---------------------------------------------------------------------------
# parse() — basic structure
# ---------------------------------------------------------------------------

def test_parse_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "empty.jsonl")
        open(path, "w").close()
        result = parse(path)
        assert result == ""


def test_parse_no_conversation_messages():
    with tempfile.TemporaryDirectory() as tmp:
        lines = [
            {"type": "mode", "mode": "normal", "sessionId": "abc"},
            {"type": "system", "content": "hello"},
        ]
        path = _write_jsonl(lines, tmp)
        result = parse(path)
        assert result == ""


def test_parse_malformed_lines_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        with open(path, "w") as f:
            f.write("NOT JSON\n")
            f.write(json.dumps(_make_user("hello")) + "\n")
        result = parse(path)
        assert "hello" in result


def test_parse_user_plain_string():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_user("startup")], tmp)
        result = parse(path)
        assert "**User**" in result
        assert "startup" in result


def test_parse_user_content_list_text_only():
    content = [{"type": "text", "text": "hello world"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_user(content)], tmp)
        result = parse(path)
        assert "hello world" in result


def test_parse_user_content_list_ignores_image():
    content = [
        {"type": "image", "source": {"type": "base64", "data": "abc"}},
        {"type": "text", "text": "visible text"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_user(content)], tmp)
        result = parse(path)
        assert "visible text" in result
        assert "base64" not in result


# ---------------------------------------------------------------------------
# parse() — tool_result rendering
# ---------------------------------------------------------------------------

def test_parse_user_tool_result_string_content():
    content = [{"type": "tool_result", "tool_use_id": "t1", "content": "3 files changed"}]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_user(content)], tmp)
        result = parse(path)
        assert "<details>" in result
        assert "tool_result" in result
        assert "3 files changed" in result


def test_parse_user_tool_result_list_content():
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [{"type": "text", "text": "stdout output"}],
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_user(content)], tmp)
        result = parse(path)
        assert "stdout output" in result
        assert "<details>" in result


def test_parse_user_tool_result_empty_omitted():
    content = [
        {"type": "tool_result", "tool_use_id": "t1", "content": ""},
        {"type": "text", "text": "still here"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_user(content)], tmp)
        result = parse(path)
        assert "<details>" not in result
        assert "still here" in result


def test_parse_user_tool_result_uses_tool_name():
    """tool_result label uses the name from the paired tool_use block."""
    assistant_msg = _make_assistant(
        [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]
    )
    user_msg = _make_user(
        [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}]
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([assistant_msg, user_msg], tmp)
        result = parse(path)
        assert "tool_result (Bash)" in result


# ---------------------------------------------------------------------------
# parse() — assistant rendering
# ---------------------------------------------------------------------------

def test_parse_assistant_text_block():
    blocks = [{"type": "text", "text": "Here is the answer."}]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        assert "**Assistant**" in result
        assert "Here is the answer." in result


def test_parse_assistant_multiple_text_blocks_joined():
    blocks = [
        {"type": "text", "text": "First paragraph."},
        {"type": "text", "text": "Second paragraph."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        assert "First paragraph." in result
        assert "Second paragraph." in result
        assert "\n\n" in result


def test_parse_assistant_tool_use_summary():
    blocks = [
        {"type": "tool_use", "name": "Read", "input": {"file_path": "/foo/bar.py"}, "id": "t1"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        assert "🔧 Read: /foo/bar.py" in result


def test_parse_assistant_thinking_block_nonempty():
    blocks = [
        {"type": "thinking", "thinking": "I need to read the file first."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        assert "<details>" in result
        assert "I need to read the file first." in result


def test_parse_assistant_thinking_block_empty_omitted():
    blocks = [
        {"type": "thinking", "thinking": ""},
        {"type": "text", "text": "Answer."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        assert "<details>" not in result
        assert "Answer." in result


def test_parse_assistant_thinking_block_null_value():
    """JSON null in thinking field must not crash (None.strip() guard)."""
    blocks = [
        {"type": "thinking", "thinking": None},
        {"type": "text", "text": "Still works."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        assert "<details>" not in result
        assert "Still works." in result


def test_parse_assistant_block_order_preserved():
    """thinking before text must render before text, not after."""
    blocks = [
        {"type": "thinking", "thinking": "Let me think."},
        {"type": "text", "text": "Here is my answer."},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl([_make_assistant(blocks)], tmp)
        result = parse(path)
        thinking_pos = result.index("<details>")
        answer_pos = result.index("Here is my answer.")
        assert thinking_pos < answer_pos


# ---------------------------------------------------------------------------
# parse() — session segmentation
# ---------------------------------------------------------------------------

def test_parse_single_session_one_header():
    lines = [_make_user("hi", session_id="sess1"), _make_assistant([{"type": "text", "text": "hello"}], session_id="sess1")]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(lines, tmp)
        result = parse(path)
        assert result.count("## Session sess1") == 1


def test_parse_multi_session_two_headers():
    lines = [
        _make_user("hi", session_id="sess1"),
        _make_user("hey", session_id="sess2"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(lines, tmp)
        result = parse(path)
        assert "## Session sess1" in result
        assert "## Session sess2" in result


def test_parse_session_header_uses_first_8_chars():
    lines = [_make_user("x", session_id="abcdefghijklmnop")]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(lines, tmp)
        result = parse(path)
        assert "## Session abcdefgh" in result


def test_parse_session_date_shown():
    lines = [_make_user("x", timestamp="2026-06-05T01:58:00.000Z")]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(lines, tmp)
        result = parse(path)
        assert "2026-06-" in result  # date rendered (day may vary by timezone)


# ---------------------------------------------------------------------------
# parse() — non-conversation lines filtered
# ---------------------------------------------------------------------------

def test_parse_filters_metadata_types():
    lines = [
        {"type": "last-prompt", "sessionId": "s1"},
        {"type": "mode", "sessionId": "s1"},
        {"type": "permission-mode", "sessionId": "s1"},
        {"type": "ai-title", "sessionId": "s1"},
        _make_user("visible"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(lines, tmp)
        result = parse(path)
        assert "visible" in result
        assert "last-prompt" not in result
