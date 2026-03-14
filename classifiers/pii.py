"""
PII redactor — detect and mask personally identifiable information in text.

Targets Taiwan-specific PII patterns:
- 身分證字號 (National ID): A123456789 → A12345****
- 統一編號 (Business ID): 12345678 → 1234****
- 手機號碼 (Mobile): 0912-345-678 → 0912-***-***
- 市話 (Landline): 02-1234-5678 → 02-****-****
- 信用卡號 (Credit card): 4321-1234-5678-9012 → 4321-****-****-9012

Design:
- Runs BEFORE AI call (prevents PII from reaching cloud APIs)
- Runs AFTER rendering (catches any PII the AI might echo back)
- Configurable via config.json "pii" section
- Disabled by default — opt-in to avoid surprising existing users
"""
import re
import logging

logger = logging.getLogger(__name__)


# --- Pattern definitions ---
# Each entry: (name, compiled regex, replacement function)
# Replacement functions receive a re.Match and return the masked string.

def _mask_national_id(m):
    """A123456789 → A12345****"""
    v = m.group(0)
    return v[:5] + "****"


def _mask_business_id(m):
    """12345678 → 1234****"""
    v = m.group(0)
    return v[:4] + "****"


def _mask_mobile(m):
    """0912-345-678 or 0912345678 → 0912-***-***"""
    v = m.group(0)
    # Normalize: extract the first 4 digits
    digits = re.sub(r"[^\d]", "", v)
    return digits[:4] + "-***-***"


def _mask_landline(m):
    """02-1234-5678 → 02-****-****"""
    v = m.group(0)
    # Extract area code (digits before first separator or first 2-3 digits)
    area = m.group("area")
    return area + "-****-****"


def _mask_credit_card(m):
    """4321-1234-5678-9012 → 4321-****-****-9012"""
    v = m.group(0)
    digits = re.sub(r"[^\d]", "", v)
    return digits[:4] + "-****-****-" + digits[-4:]


# Pattern registry: (name, regex, mask_func)
# Order matters — more specific patterns first to avoid false positives.
_PATTERNS = [
    (
        "credit_card",
        re.compile(
            r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
        ),
        _mask_credit_card,
    ),
    (
        "national_id",
        re.compile(
            # Taiwan national ID: 1 uppercase letter + [1-2] + 8 digits
            r"\b[A-Z][12]\d{8}\b"
        ),
        _mask_national_id,
    ),
    (
        "mobile",
        re.compile(
            # Taiwan mobile: 09xx-xxx-xxx or 09xxxxxxxx
            r"\b09\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b"
        ),
        _mask_mobile,
    ),
    (
        "landline",
        re.compile(
            # Taiwan landline: 0x-xxxx-xxxx or 0xx-xxx-xxxx
            r"\b(?P<area>0[2-9]\d?)[\s\-]\d{3,4}[\s\-]\d{4}\b"
        ),
        _mask_landline,
    ),
    (
        "business_id",
        re.compile(
            # Taiwan unified business number: exactly 8 digits, standalone
            # Use lookaround to avoid matching parts of longer numbers
            r"(?<!\d)\d{8}(?!\d)"
        ),
        _mask_business_id,
    ),
]


def redact(text, enabled_patterns=None):
    """
    Redact PII from text using configured patterns.

    Args:
        text: input text to redact
        enabled_patterns: list of pattern names to apply, or None for all.
                          Valid names: national_id, business_id, mobile,
                          landline, credit_card

    Returns:
        (redacted_text, redaction_count) — count is total number of redactions made
    """
    if not text:
        return text, 0

    total = 0
    for name, pattern, mask_fn in _PATTERNS:
        if enabled_patterns is not None and name not in enabled_patterns:
            continue

        matches = list(pattern.finditer(text))
        if matches:
            text = pattern.sub(mask_fn, text)
            total += len(matches)
            logger.debug(f"PII redacted: {len(matches)}x {name}")

    if total > 0:
        logger.info(f"PII redaction: {total} item(s) masked")

    return text, total
