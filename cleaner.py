#!/usr/bin/env python3
"""
doc-cleaner — Convert PDF, DOCX, XLSX, and text files to clean, structured Markdown.

CJK-friendly. Table-friendly. Privacy-first.

Part of the notoriouslab open-source toolkit.
"""
import os
import re
import sys
import json
import argparse
import logging
import tempfile
from pathlib import Path

__version__ = "1.0.2"

logger = logging.getLogger("doc-cleaner")

# Exit codes
EXIT_OK = 0             # all files processed successfully
EXIT_PARTIAL = 1        # some files failed
EXIT_NO_INPUT = 2       # no processable files found or config error

# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md"}

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(config_path):
    """Load JSON config, return empty dict if not found."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_prompt(config, config_path=None):
    """Load the AI prompt template from config or default."""
    prompt_path = config.get("ai", {}).get("prompt_template")
    if prompt_path and not os.path.isabs(prompt_path):
        # Try relative to config dir first, then script dir
        candidates = []
        if config_path:
            candidates.append(os.path.join(os.path.dirname(config_path), prompt_path))
        candidates.append(os.path.join(SCRIPT_DIR, prompt_path))

        resolved = None
        for c in candidates:
            if os.path.exists(c):
                resolved = c
                break

        if resolved:
            with open(resolved, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning(f"Prompt template not found: {prompt_path}, using default")
    elif prompt_path and os.path.isabs(prompt_path):
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning(f"Prompt template not found: {prompt_path}, using default")

    # Default prompt
    default_path = os.path.join(SCRIPT_DIR, "prompts", "default.txt")
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Analyze this document and output JSON with keys: title, summary, refined_markdown, tags."


def warn_config_secrets(config):
    """Warn if config.json contains fields that should be in .env instead."""
    secret_paths = [
        (["ai", "gemini", "api_key"], "GEMINI_API_KEY"),
        (["ai", "groq", "api_key"], "GROQ_API_KEY"),
        (["ai", "ollama", "api_key"], "OLLAMA_API_KEY"),
        (["pdf", "password"], "PDF_PASSWORD"),
    ]
    for keys, env_name in secret_paths:
        obj = config
        for k in keys:
            obj = obj.get(k, {}) if isinstance(obj, dict) else {}
        if obj and isinstance(obj, str):
            logger.warning(
                f"⚠️  Secret found in config.json ({'.'.join(keys)}). "
                f"Move it to .env as {env_name} and remove from config.json. "
                f"config.json may be accidentally committed to git."
            )


def validate_patterns(config):
    """Pre-validate ad_truncation_patterns regex at startup."""
    patterns = config.get("ad_truncation_patterns", [])
    for i, pat in enumerate(patterns):
        try:
            re.compile(pat)
        except re.error as e:
            logger.error(f"Invalid regex in ad_truncation_patterns[{i}]: {pat!r} — {e}")
            sys.exit(EXIT_NO_INPUT)


def create_ai_backend(ai_mode, config):
    """Create the appropriate AI backend based on config."""
    if ai_mode == "none":
        return None

    ai_config = config.get("ai", {})

    if ai_mode == "gemini":
        try:
            from ai.gemini import GeminiBackend
        except ImportError:
            logger.error(
                "Gemini backend requires google-genai. "
                "Install with: pip install google-genai python-dotenv"
            )
            sys.exit(EXIT_NO_INPUT)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error(
                "GEMINI_API_KEY not set. Add it to your .env file:\n"
                "  echo 'GEMINI_API_KEY=your-key-here' >> .env\n"
                "Do NOT put API keys in config.json — it may be committed to git."
            )
            sys.exit(EXIT_NO_INPUT)
        model = ai_config.get("gemini", {}).get("model", "gemini-2.5-pro")
        return GeminiBackend(api_key=api_key, model=model)

    if ai_mode == "groq":
        from ai.groq import GroqBackend

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error(
                "GROQ_API_KEY not set. Add it to your .env file:\n"
                "  echo 'GROQ_API_KEY=your-key-here' >> .env\n"
                "Do NOT put API keys in config.json — it may be committed to git."
            )
            sys.exit(EXIT_NO_INPUT)

        groq_config = ai_config.get("groq", {})
        model = groq_config.get("model", "meta-llama/llama-4-scout-17b-16e-instruct")
        base_url = groq_config.get("base_url", "https://api.groq.com/openai/v1")
        timeout = groq_config.get("timeout", 120)
        return GroqBackend(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    if ai_mode == "ollama":
        try:
            from ai.ollama import OllamaBackend
        except ImportError:
            logger.error(
                "Ollama backend requires the ollama package. "
                "Install with: pip install ollama\n"
                "Also ensure Ollama is running: https://ollama.com"
            )
            sys.exit(EXIT_NO_INPUT)

        ollama_config = ai_config.get("ollama", {})
        model = ollama_config.get("model", "qwen3.5:9b")
        host = ollama_config.get("host", "http://localhost:11434")
        vision_models = ollama_config.get("vision_models")
        return OllamaBackend(model=model, host=host, vision_models=vision_models)

    logger.error(f"Unknown AI backend: {ai_mode}")
    sys.exit(EXIT_NO_INPUT)


def parse_file(filepath, config):
    """
    Parse a document file and return extracted text + optional images.

    Returns: (text, images)
        - text: extracted text string (may be empty for scanned PDFs)
        - images: list of PIL.Image objects (for PDF vision mode), or None
    """
    ext = os.path.splitext(filepath)[1].lower()
    pdf_config = config.get("pdf", {})
    images = None
    text = ""

    if ext == ".pdf":
        try:
            from parsers import pdf
            from classifiers.pdf_classifier import classify, PdfType
            from classifiers.noise import clean_text
        except ImportError as e:
            raise ImportError(
                f"PDF processing requires PyMuPDF. Install with: pip install -r requirements.txt\n"
                f"  Missing: {e.name or e}"
            )

        # Decrypt if needed
        password = pdf_config.get("password") or os.getenv("PDF_PASSWORD")
        target = filepath
        if password:
            output_dir = pdf_config.get("decrypt_dir")
            decrypted = pdf.decrypt_pdf(filepath, password=password, output_dir=output_dir)
            if decrypted:
                target = decrypted

        # Classify and extract
        pdf_type, raw_text, metadata = classify(target)

        cutoff_patterns = config.get("ad_truncation_patterns")
        strip_urls = config.get("strip_urls", True)
        text = clean_text(raw_text, cutoff_patterns=cutoff_patterns, strip_urls=strip_urls)

        if pdf_type in (PdfType.SCANNED, PdfType.LAYOUT_BROKEN):
            dpi = pdf_config.get("dpi", 200)
            max_pages = pdf_config.get("max_pages", 15)
            images = pdf.extract_images(target, dpi=dpi, max_pages=max_pages)
            if not images and not text:
                logger.warning(f"No text or images extracted from {os.path.basename(filepath)}")

    elif ext == ".docx":
        try:
            from parsers.docx import parse
        except ImportError as e:
            raise ImportError(
                f"DOCX processing requires python-docx. Install with: pip install python-docx\n"
                f"  Missing: {e.name or e}"
            )
        text = parse(filepath)

    elif ext in (".xlsx", ".xls", ".csv"):
        try:
            from parsers.xlsx import parse
        except ImportError as e:
            raise ImportError(
                f"Spreadsheet processing requires pandas + openpyxl. "
                f"Install with: pip install pandas openpyxl\n"
                f"  Missing: {e.name or e}"
            )
        text = parse(filepath)

    elif ext in (".txt", ".md"):
        from parsers.text import parse
        text = parse(filepath)

    else:
        logger.warning(f"Unsupported file type: {ext}")

    return text, images


def process_file(filepath, ai_backend, prompt, config, output_dir, dry_run=False):
    """
    Process a single file: parse → (optional AI) → Markdown output.

    Returns: (status, output_path)
        - status: "ok" | "dry_run" | "no_content" | "write_error" | "error"
        - output_path: path to output file, or None on failure
    """
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{stem}.md")

    # Avoid overwriting existing output from a different source file
    if os.path.exists(output_path):
        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{stem}_{counter}.md")):
            counter += 1
        output_path = os.path.join(output_dir, f"{stem}_{counter}.md")
        logger.info(f"  Output collision: {stem}.md exists, using {stem}_{counter}.md")

    logger.info(f"Processing: {filename}")

    if dry_run:
        ext = os.path.splitext(filepath)[1].lower()
        logger.info(f"  [dry-run] Would process {filename} ({ext}) → {output_path}")
        return "dry_run", output_path

    try:
        text, images = parse_file(filepath, config)

        if not text and not images:
            logger.warning(f"  No content extracted from {filename}")
            return "no_content", None

        # PII redaction (opt-in via config)
        pii_config = config.get("pii", {})
        pii_enabled = pii_config.get("enabled", False)
        pii_patterns = pii_config.get("patterns", None)  # None = all patterns

        if pii_enabled and text:
            from classifiers.pii import redact as redact_pii
            text, pii_count = redact_pii(text, enabled_patterns=pii_patterns)
            if pii_count:
                logger.info(f"  PII: {pii_count} item(s) redacted before processing")

        frontmatter = config.get("output", {}).get("frontmatter", True)

        if ai_backend:
            # AI mode: send to LLM for structuring
            from ai.base import clean_json_response
            from output.markdown import render_ai_output, render_raw_output

            # Retry once on transient errors (429/503/timeout) before fallback
            max_retries = config.get("ai", {}).get("max_retries", 1)
            raw_response = None
            last_err = None
            for attempt in range(1 + max_retries):
                try:
                    raw_response = ai_backend.call(prompt=prompt, images=images, text=text)
                    break
                except Exception as ai_err:
                    last_err = ai_err
                    if attempt < max_retries:
                        import time
                        wait = 2 ** attempt  # 1s, 2s, ...
                        logger.warning(
                            f"  AI call failed ({ai_err}), retrying in {wait}s "
                            f"(attempt {attempt + 1}/{1 + max_retries})"
                        )
                        time.sleep(wait)

            if raw_response is None:
                # All retries exhausted — fall back to raw mode if we have text
                if text:
                    logger.warning(
                        f"  AI call failed after {1 + max_retries} attempts ({last_err}) "
                        f"— falling back to raw mode"
                    )
                    content = render_raw_output(
                        text, filename, source_path=filename,
                        frontmatter=frontmatter,
                    )
                else:
                    raise last_err  # no text to fall back on, propagate error
            else:
                data = clean_json_response(raw_response)

                # Graceful degradation: if JSON repair failed badly, fall back to raw mode
                if data.get("status") == "partial_recovery" and text:
                    logger.warning(f"  AI JSON output corrupted — falling back to raw mode")
                    content = render_raw_output(
                        text, filename, source_path=filename,
                        frontmatter=frontmatter,
                    )
                else:
                    content = render_ai_output(
                        data, filename, source_path=filename,
                        frontmatter=frontmatter,
                    )
        else:
            # Raw mode: output extracted text directly
            from output.markdown import render_raw_output

            content = render_raw_output(
                text, filename, source_path=filename,
                frontmatter=frontmatter,
            )

        # Final PII sweep on rendered output (catches AI-echoed PII)
        if pii_enabled:
            from classifiers.pii import redact as redact_pii
            content, pii_output_count = redact_pii(content, enabled_patterns=pii_patterns)
            if pii_output_count:
                logger.info(f"  PII: {pii_output_count} item(s) redacted from output")

        # Atomic write via tempfile (safe for parallel invocations)
        try:
            os.makedirs(output_dir, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                os.replace(tmp_path, output_path)
            except BaseException:
                # Clean up temp file on any failure (e.g. cross-filesystem rename)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.error(f"  Write failed for {output_path}: {e}")
            return "write_error", None

        logger.info(f"  → {output_path}")
        return "ok", output_path

    except Exception as e:
        logger.error(f"  Failed: {filename}: {e}")
        return "error", None


def collect_files(input_path):
    """Collect processable files from a path (file or directory)."""
    if os.path.isfile(input_path):
        # Security: resolve symlinks to prevent directory traversal
        real_path = os.path.realpath(input_path)
        if os.path.islink(input_path):
            logger.info(f"Resolved symlink: {input_path} → {real_path}")
        ext = os.path.splitext(real_path)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            return [real_path]
        else:
            logger.warning(f"Unsupported file type: {input_path}")
            return []

    if os.path.isdir(input_path):
        real_root = os.path.realpath(input_path)
        files = []
        skipped_dirs = []
        for f in sorted(os.listdir(input_path)):
            fp = os.path.realpath(os.path.join(input_path, f))
            # P4 security: reject symlinks escaping the input directory
            if not fp.startswith(real_root + os.sep) and fp != real_root:
                logger.warning(f"Skipping symlink escape: {f}")
                continue
            if os.path.isdir(fp):
                skipped_dirs.append(f)
            elif os.path.isfile(fp) and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS:
                files.append(fp)
        if skipped_dirs:
            logger.debug(
                f"Skipped {len(skipped_dirs)} subdirectory(ies) (non-recursive): "
                + ", ".join(skipped_dirs)
            )
        return files

    logger.error(f"Input not found: {input_path}")
    return []


def main():
    if sys.version_info < (3, 9):
        sys.exit("doc-cleaner requires Python 3.9+. Current: " + sys.version)

    parser = argparse.ArgumentParser(
        description="doc-cleaner — Convert documents to clean, structured Markdown.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python cleaner.py --input statement.pdf\n"
            "  python cleaner.py --input ./downloads/ --ai none\n"
            "  python cleaner.py --input report.xlsx --dry-run --verbose\n"
            "\n"
            "Part of the notoriouslab pipeline:\n"
            "  gmail-statement-fetcher → doc-cleaner → personal-cfo\n"
        ),
    )
    parser.add_argument("--input", "-i", required=True, help="file or directory to process")
    parser.add_argument("--output-dir", "-o", default="./output", help="output directory (default: ./output)")
    parser.add_argument("--config", default=None, help="path to config JSON (default: <script-dir>/config.json)")
    parser.add_argument("--ai", choices=["gemini", "groq", "ollama", "none"], default=None, help="AI backend (default: from config or gemini)")
    parser.add_argument("--password", default=None, help="PDF decryption password (overrides .env and config)")
    parser.add_argument("--summary", action="store_true", help="print JSON summary to stdout after processing")
    parser.add_argument("--dry-run", action="store_true", help="preview without writing files")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"doc-cleaner {__version__}")

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Config
    config_path = args.config or os.path.join(SCRIPT_DIR, "config.json")
    config = load_config(config_path)

    # Security: warn if secrets leaked into config.json
    warn_config_secrets(config)

    # Validate regex patterns at startup
    validate_patterns(config)

    # PDF password priority: CLI > .env > config.json
    if args.password:
        if len(args.password) > 1024:
            logger.error("--password too long (max 1024 chars)")
            sys.exit(EXIT_NO_INPUT)
        config.setdefault("pdf", {})["password"] = args.password

    # AI mode priority: CLI --ai > config.json > default "gemini"
    ai_mode = args.ai or config.get("ai", {}).get("backend", "gemini")

    # AI backend
    ai_backend = create_ai_backend(ai_mode, config)
    prompt = load_prompt(config, config_path=config_path) if ai_backend else None

    # Collect files
    files = collect_files(args.input)
    if not files:
        logger.error("No processable files found.")
        sys.exit(EXIT_NO_INPUT)

    logger.info(f"doc-cleaner v{__version__} — {len(files)} file(s) to process")
    if args.dry_run:
        logger.info("[DRY RUN] No files will be written.")

    # Process
    results = []
    for filepath in files:
        status, output_path = process_file(
            filepath, ai_backend, prompt, config, args.output_dir, dry_run=args.dry_run,
        )
        results.append({
            "file": os.path.basename(filepath),
            "output": os.path.relpath(output_path) if output_path else None,
            "status": status,
        })

    success = sum(1 for r in results if r["status"] in ("ok", "dry_run"))
    logger.info(f"Done: {success}/{len(files)} files processed.")

    # Machine-readable summary for AI agents and scripts
    if args.summary:
        summary = {
            "version": __version__,
            "total": len(files),
            "success": success,
            "failed": len(files) - success,
            "files": results,
        }
        print(json.dumps(summary, ensure_ascii=False))

    sys.exit(EXIT_OK if success == len(files) else EXIT_PARTIAL)


if __name__ == "__main__":
    main()
