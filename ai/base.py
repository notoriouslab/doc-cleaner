"""
Abstract AI backend interface.

All AI backends (Gemini, Ollama, etc.) implement this interface so the
main cleaner can swap backends via config without changing code.

Usage:
    backend = GeminiBackend(api_key="...", model="gemini-2.5-pro")
    result = backend.call(prompt="Analyze this document", images=[pil_img], text="...")
"""
import ipaddress
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Optional
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


class _BlockedNetworkError(ValueError):
    """Raised when a base_url resolves to a private/internal network."""


def _is_blocked_addr(addr) -> bool:
    """Return True if addr (or its IPv4-mapped equivalent) is in any blocked range."""
    mapped = getattr(addr, "ipv4_mapped", None)
    check = mapped if mapped is not None else addr
    return (
        check.is_private
        or check.is_loopback
        or check.is_link_local
        or check.is_unspecified
        or check.is_multicast
    )


def _try_parse_nonstandard_ip(host: str):
    """
    Try to interpret host as a non-standard numeric IP form that OS resolvers accept.

    Handles:
    - Decimal integer: "2130706433" → 127.0.0.1
    - Hex: "0x7f000001" → 127.0.0.1
    - Octal: "0177" → 127.0.0.0-ish
    - Partial dotted (2/3 parts): "127.1" → 127.0.0.1

    Returns an IPv4Address on success, or None if host is a domain name.
    """
    import socket as _socket
    # socket.inet_aton handles decimal integers and partial-dotted notation on POSIX
    try:
        return ipaddress.IPv4Address(_socket.inet_aton(host))
    except (_socket.error, OSError, ValueError):
        pass
    # int() with auto-base handles 0x hex and 0-prefixed octal
    try:
        n = int(host, 0)
        if 0 <= n <= 0xFFFFFFFF:
            return ipaddress.IPv4Address(n)
    except (ValueError, OverflowError):
        pass
    return None


def validate_base_url(url: str, backend_name: str = "AI backend") -> str:
    """
    Validate that a base_url does not point to private/internal networks.

    Uses ipaddress for standard IP literals and also catches non-standard numeric
    forms (decimal integers, hex, partial notation) that OS resolvers may accept.

    Raises ValueError for any blocked address; returns the normalised URL on success.
    """
    parsed = urlparse(url.rstrip("/"))
    if parsed.scheme not in ("https", "http"):
        raise ValueError(
            f"{backend_name} base_url must use http(s), got {parsed.scheme!r}"
        )

    host = parsed.hostname or ""

    # Empty host (e.g. "http://") resolves to loopback on many platforms
    if not host:
        raise _BlockedNetworkError(
            f"{backend_name} base_url must not use an empty host."
        )

    # urllib.request.Request._parse() calls unquote() on the host before connecting,
    # so "127%2E0%2E0%2E1" is decoded to "127.0.0.1" at connection time.
    # Decode here so validation sees the same host that the OS resolver will receive.
    host = unquote(host)

    if host.lower() in _BLOCKED_HOSTNAMES:
        raise _BlockedNetworkError(
            f"{backend_name} base_url must not point to private/internal networks, "
            f"got {host!r}. If you need local inference, use Ollama instead."
        )

    try:
        addr = ipaddress.ip_address(host)
        if _is_blocked_addr(addr):
            raise _BlockedNetworkError(
                f"{backend_name} base_url must not point to private/internal networks, "
                f"got {host!r}. If you need local inference, use Ollama instead."
            )
    except _BlockedNetworkError:
        raise
    except ValueError:
        # host is not a standard IP literal — check non-standard numeric forms that
        # OS resolvers (getaddrinfo/inet_aton) may still interpret as private IPs
        alt = _try_parse_nonstandard_ip(host)
        if alt is not None and _is_blocked_addr(alt):
            raise _BlockedNetworkError(
                f"{backend_name} base_url must not point to private/internal networks, "
                f"got {host!r}. If you need local inference, use Ollama instead."
            )
        # Otherwise it's a domain name — allow

    return parsed.geturl()


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

    # Try direct parse first — avoid unnecessary repair that may corrupt valid JSON
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

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
