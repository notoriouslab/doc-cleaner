"""
Abstract AI backend interface.

All AI backends (Gemini, Ollama, etc.) implement this interface so the
main cleaner can swap backends via config without changing code.

Usage:
    backend = GeminiBackend(api_key="...", model="gemini-2.5-pro")
    result = backend.call(prompt="Analyze this document", images=[pil_img], text="...")
"""
from abc import ABC, abstractmethod
from typing import Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


class AIBackend(ABC):
    """Abstract base class for AI backends."""

    @abstractmethod
    def call(self, prompt: str, images: Optional[list] = None, text: Optional[str] = None) -> str:
        """
        Send a prompt (with optional images and text) to the AI model.

        Args:
            prompt: the system/instruction prompt
            images: optional list of PIL.Image objects (for vision mode)
            text: optional extracted text content

        Returns:
            raw response string from the model
        """
        ...


def clean_json_response(raw_text):
    """
    Parse AI response as JSON with auto-repair for common LLM quirks.

    Handles:
    - ```json fencing removal
    - Trailing comma removal
    - Unterminated string/object closure
    - Fallback regex extraction for refined_markdown field
    """
    s = raw_text.strip()

    # Remove markdown code fencing
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()

    # Auto-fix trailing commas
    s = re.sub(r",\s*([\]}])", r"\1", s)

    # Auto-fix unterminated structures (balance brackets before closing)
    if not (s.endswith("}") or s.endswith("]")):
        # Close any unterminated string — count quotes (ignoring escaped ones)
        unescaped_quotes = len(re.findall(r'(?<!\\)"', s))
        if unescaped_quotes % 2 != 0:
            s += '"'
        # Balance unclosed brackets in correct nesting order using a stack
        stack = []
        in_string = False
        prev_char = ""
        for ch in s:
            if ch == '"' and prev_char != "\\":
                in_string = not in_string
            elif not in_string:
                if ch in ("{", "["):
                    stack.append(ch)
                elif ch == "}" and stack and stack[-1] == "{":
                    stack.pop()
                elif ch == "]" and stack and stack[-1] == "[":
                    stack.pop()
            prev_char = ch
        # Close in reverse order (innermost first)
        for opener in reversed(stack):
            s += "}" if opener == "{" else "]"

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Heavy rescue: extract refined_markdown via regex
        match = re.search(
            r'"refined_markdown"\s*:\s*"((?:[^"\\]|\\.)*)"',
            s,
            re.IGNORECASE | re.DOTALL,
        )
        markdown = match.group(1).replace("\\n", "\n") if match else raw_text
        return {
            "summary": "JSON parse error — recovered raw content",
            "refined_markdown": markdown,
            "status": "partial_recovery",
        }
