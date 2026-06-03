"""
Tests for validate_base_url() — SSRF allowlist using ipaddress module.

Covers every bypass that existed in the old string-prefix blacklist.
"""
import pytest
from ai.base import validate_base_url


# ── URLs that MUST be rejected ────────────────────────────────────────────────

BLOCKED = [
    # Classic loopback hostnames
    ("http://localhost/v1",               "loopback hostname"),
    ("http://localhost.localdomain/v1",   "loopback alias"),
    # Full 127/8 loopback range (old code only blocked 127.0.0.1)
    ("http://127.0.0.1/v1",              "loopback 127.0.0.1"),
    ("http://127.0.0.2/v1",              "loopback 127.0.0.2 — old code missed this"),
    ("http://127.255.255.255/v1",        "loopback upper end"),
    # IPv6 loopback
    ("http://[::1]/v1",                  "IPv6 loopback ::1"),
    # IPv4-mapped IPv6 loopback (old code missed this)
    ("http://[::ffff:127.0.0.1]/v1",     "IPv4-mapped IPv6 loopback"),
    # Unspecified address (old code missed 0.0.0.0)
    ("http://0.0.0.0/v1",               "INADDR_ANY — old code missed this"),
    # RFC 1918 private ranges
    ("http://10.0.0.1/v1",              "RFC1918 10.x"),
    ("http://192.168.1.100/v1",         "RFC1918 192.168.x"),
    ("http://172.16.0.1/v1",            "RFC1918 172.16 — lower edge"),
    ("http://172.31.255.255/v1",        "RFC1918 172.31 — upper edge"),
    # IPv6 ULA (fc00::/7) — old code missed entirely
    ("http://[fd00::1]/v1",             "IPv6 ULA — old code missed this"),
    ("http://[fc00::1]/v1",             "IPv6 ULA fc00 — old code missed this"),
    # Link-local
    ("http://169.254.1.1/v1",           "link-local"),
    ("http://[fe80::1]/v1",             "IPv6 link-local"),
    # Non-standard numeric IP forms that OS resolvers accept (round 2 findings)
    ("http://2130706433/v1",            "decimal int 127.0.0.1"),
    ("http://0x7f000001/v1",            "hex 127.0.0.1"),
    ("http:///v1",                      "empty host — resolves to loopback on many platforms"),
    # Percent-encoded bypass: urllib.request unquotes the host before connecting
    # but urlparse.hostname returns the encoded form (round 3 findings)
    ("http://127%2E0%2E0%2E1/v1",      "percent-encoded loopback dots"),
    ("http://%31%32%37%2e%30%2e%30%2e%31/v1", "fully percent-encoded 127.0.0.1"),
    ("http://0%2E0%2E0%2E0/v1",        "percent-encoded 0.0.0.0"),
    ("http://10%2E0%2E0%2E1/v1",       "percent-encoded RFC1918 10.x"),
    ("http://192%2E168%2E1%2E1/v1",    "percent-encoded RFC1918 192.168.x"),
    # Wrong scheme
    ("ftp://api.groq.com/v1",           "non-http scheme"),
    ("file:///etc/passwd",              "file scheme"),
]


@pytest.mark.parametrize("url,reason", BLOCKED)
def test_blocked(url, reason):
    """Blocked URL raises ValueError."""
    with pytest.raises(ValueError, match="must not point|must use|must not use"):
        validate_base_url(url, backend_name="Test")


# ── URLs that MUST be allowed ─────────────────────────────────────────────────

ALLOWED = [
    "https://api.groq.com/openai/v1",
    "https://integrate.api.nvidia.com/v1",
    "http://my-proxy.internal.company.example.com/v1",  # hostname, not IP
    "https://1.2.3.4/v1",                               # public IP
    "https://8.8.8.8/v1",                               # public IP (Google DNS)
    # 172.15.x and 172.32.x are NOT RFC 1918 — they are public addresses
    # (RFC 1918 only covers 172.16.0.0/12 = 172.16-31.x.x)
    "http://172.15.0.1/v1",
    "http://172.32.0.1/v1",
]


@pytest.mark.parametrize("url", ALLOWED)
def test_allowed(url):
    """Allowed URL returns normalised URL without raising."""
    result = validate_base_url(url, backend_name="Test")
    assert result.startswith("http")


# ── Backend integration: groq + nvidia use validate_base_url ─────────────────

def test_groq_backend_rejects_localhost():
    """GroqBackend.__init__ raises on localhost base_url."""
    from ai.groq import GroqBackend
    with pytest.raises(ValueError, match="must not point"):
        GroqBackend(api_key="k", base_url="http://localhost/v1")


def test_groq_backend_rejects_0000():
    """GroqBackend.__init__ raises on 0.0.0.0 (old blacklist missed this)."""
    from ai.groq import GroqBackend
    with pytest.raises(ValueError, match="must not point"):
        GroqBackend(api_key="k", base_url="http://0.0.0.0/v1")


def test_nvidia_backend_rejects_ipv6_ula():
    """NvidiaBackend.__init__ raises on IPv6 ULA (old blacklist missed this)."""
    from ai.nvidia import NvidiaBackend
    with pytest.raises(ValueError, match="must not point"):
        NvidiaBackend(api_key="k", base_url="http://[fd00::1]/v1")


def test_groq_backend_allows_172_15():
    """GroqBackend allows 172.15.x — it is NOT in RFC 1918 (172.16.0.0/12 = 172.16-31)."""
    from ai.groq import GroqBackend
    # Should not raise — 172.15.x is a public address
    backend = GroqBackend(api_key="k", base_url="http://172.15.0.1/v1")
    assert "172.15.0.1" in backend._base_url
