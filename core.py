"""Shared conversion API — used by both CLI and GUI."""
from pathlib import Path
import logging

from cleaner import load_config, load_prompt, create_ai_backend, process_file, SCRIPT_DIR

_STATUS_MAP = {
    "ok": "ok",
    "dry_run": "dry_run",   # preserved so CLI --summary is not a breaking change
    "no_content": "skipped",
    "write_error": "error",
    "error": "error",
}


class _ErrorCapture(logging.Handler):
    """Captures warning-and-above messages from conversion-related loggers."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _build_env(ai="none", config_path=None):
    """Load config + create backend + load prompt. Shared setup for GUI and batch calls."""
    if config_path is None:
        config_path = str(SCRIPT_DIR / "config.json")
    config = load_config(config_path)
    ai_backend = create_ai_backend(ai, config)
    prompt = load_prompt(config, config_path=config_path) if ai_backend else None
    return config, ai_backend, prompt


def _run_one(filepath, ai_backend, prompt, config, output_dir, dry_run=False):
    """
    Call process_file and return a GUI-shaped result dict.
    Attaches to doc-cleaner, parsers, and classifiers loggers to capture
    warning-and-above messages emitted during conversion.
    """
    capture = _ErrorCapture()
    _watched = [
        logging.getLogger("doc-cleaner"),
        logging.getLogger("parsers"),
        logging.getLogger("classifiers"),
    ]
    for _log in _watched:
        _log.addHandler(capture)
    try:
        status_raw, out_path = process_file(
            filepath, ai_backend, prompt, config, output_dir, dry_run=dry_run
        )
    finally:
        for _log in _watched:
            _log.removeHandler(capture)

    status = _STATUS_MAP.get(status_raw, "error")
    if status == "error":
        error_msg = capture.messages[-1] if capture.messages else "轉換失敗"
    elif status == "skipped":
        error_msg = "未擷取到文字內容"
    else:
        error_msg = None

    return {
        "file": Path(filepath).name,
        "input": str(Path(filepath).resolve()),
        "output": str(Path(out_path).resolve()) if out_path else None,
        "status": status,
        "error": error_msg,
    }


def convert_file(input_path, output_dir=None, ai="none"):
    """
    Convert a single file. Loads config, builds backend, runs extraction.

    Returns: {file, input, output, status, error}
        status: "ok" | "skipped" | "error"
        output_dir=None writes the result beside the source file.
    """
    input_path = str(Path(input_path).resolve())
    if output_dir is None:
        output_dir = str(Path(input_path).parent)
    config, ai_backend, prompt = _build_env(ai=ai)
    return _run_one(input_path, ai_backend, prompt, config, output_dir)


def convert_files(paths, output_resolver=None, ai="none", config=None, config_path=None, dry_run=False):
    """
    Batch conversion. Builds config + backend + prompt once, reuses across all files.
    Continues past failing files — one error never aborts the batch.

    paths: iterable of file path strings
    output_resolver: callable(abspath) -> output_dir string; None → sibling of each source
    ai: AI backend name ("none" for pure extraction)
    config: pre-processed config dict (optional; None loads default config.json)
    config_path: path to config file (for prompt resolution when config is provided)
    dry_run: skip writing files, preview only

    Returns: list of {file, input, output, status, error} dicts.
    """
    if config is None:
        config, ai_backend, prompt = _build_env(ai=ai, config_path=config_path)
    else:
        ai_backend = create_ai_backend(ai, config)
        prompt = load_prompt(config, config_path=config_path) if ai_backend else None

    results = []
    for path in paths:
        path = str(Path(path).resolve())
        output_dir = output_resolver(path) if output_resolver else str(Path(path).parent)
        results.append(_run_one(path, ai_backend, prompt, config, output_dir, dry_run=dry_run))
    return results
