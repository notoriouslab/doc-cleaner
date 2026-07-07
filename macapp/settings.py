"""
GUI preferences persistence (Phase A, D1).

A small, dependency-free settings layer for the desktop app. Stores user
preferences as JSON in the platform's writable application-data directory —
never inside the read-only signed `.app` bundle.

This module MUST NOT import `webview`: it is pure logic so it can be unit
tested without launching a window. All webview interaction lives in app.py.

Schema (versioned for forward migration):

    {
        "version": 1,
        "output_mode": "sibling",      # "sibling" | "desktop" | "custom"
        "custom_output_dir": null,     # absolute path when output_mode == "custom"
        "last_input_dir": null         # dir of the most recently picked/dropped source
    }

Robustness contract: a missing or malformed settings file falls back to
defaults and never raises, so a corrupt file can never stop the app from
launching or converting.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

APP_NAME = "Doc Cleaner"
SETTINGS_FILENAME = "settings.json"
SCHEMA_VERSION = 1

# Canonical defaults. load() returns this shape with valid stored values merged in.
DEFAULTS = {
    "version": SCHEMA_VERSION,
    "output_mode": "sibling",
    "custom_output_dir": None,
    "last_input_dir": None,
    "output_format": "md",
}

# Per-key validators: a stored value is accepted only if it passes, else the
# default for that key is used (never raises).
_VALID_OUTPUT_MODES = {"sibling", "desktop", "custom"}
_VALID_OUTPUT_FORMATS = {"md", "epub", "both"}


def _valid(key, value):
    if key == "version":
        return isinstance(value, int)
    if key == "output_mode":
        # isinstance guard first: a non-str (e.g. dict/list) value would make
        # `value in <set>` raise TypeError (unhashable). Short-circuit instead.
        return isinstance(value, str) and value in _VALID_OUTPUT_MODES
    if key == "output_format":
        return isinstance(value, str) and value in _VALID_OUTPUT_FORMATS
    if key in ("custom_output_dir", "last_input_dir"):
        return value is None or isinstance(value, str)
    return False


def app_data_dir():
    """Return the per-OS writable application-data directory (created on demand by save())."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        base = Path.home() / ".config"
    return str(base / APP_NAME)


def _settings_path():
    return os.path.join(app_data_dir(), SETTINGS_FILENAME)


def load():
    """Return DEFAULTS merged with valid stored keys. Never raises."""
    result = dict(DEFAULTS)
    path = _settings_path()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            stored = json.load(fh)
    except FileNotFoundError:
        return result
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Settings file unreadable (%s); using defaults: %s", path, exc)
        return result

    if not isinstance(stored, dict):
        logger.warning("Settings root is not an object; using defaults")
        return result

    # Defense in depth: validation must honor the never-raise contract even if
    # a value has an unexpected type — fall back to defaults for the bad key.
    try:
        for key in DEFAULTS:
            if key in stored and _valid(key, stored[key]):
                result[key] = stored[key]
            elif key in stored:
                logger.warning("Ignoring invalid settings value for %r; using default", key)
    except Exception as exc:  # pragma: no cover — _valid is already type-safe
        logger.warning("Settings validation failed (%s); using defaults", exc)
        return dict(DEFAULTS)
    return result


def save(settings):
    """Atomically write the known-key settings dict. Best-effort; never raises to the caller."""
    directory = app_data_dir()
    payload = {key: settings.get(key, DEFAULTS[key]) for key in DEFAULTS}
    try:
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".settings-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, os.path.join(directory, SETTINGS_FILENAME))
        except Exception:
            # Clean up the temp file on any write/replace failure.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as exc:
        logger.warning("Could not save settings to %s: %s", directory, exc)
