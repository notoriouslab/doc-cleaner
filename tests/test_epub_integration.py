"""End-to-end format-selection tests (integrate-epub-output).

Covers: the JSONL write-path regression (NameError from the PR's rename),
format selection md/epub/both across the normal and JSONL paths, partial
write failure, and the `_run_one` preview field contract.
"""
import json
import os
import zipfile

import pytest

from cleaner import load_config, process_file


def _write_jsonl(tmp_path):
    path = os.path.join(str(tmp_path), "session.jsonl")
    line = {
        "type": "user",
        "sessionId": "abc123",
        "timestamp": "2026-06-05T01:58:19.000Z",
        "message": {"role": "user", "content": "測試訊息內容"},
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def _write_txt(tmp_path, name="doc.txt"):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 標題\n\n內容段落，含表格：\n\n|欄|值|\n|---|---|\n|a|1|\n")
    return path


@pytest.fixture
def config():
    return load_config(None)


class TestJsonlRegression:
    def test_jsonl_md_end_to_end(self, tmp_path, config):
        """The PR renamed output_path→primary_path but left the JSONL branch
        on the old name — every JSONL conversion raised NameError."""
        src = _write_jsonl(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        status, out_path = process_file(src, None, None, config, out_dir)
        assert status == "ok"
        assert out_path.endswith(".md") and os.path.exists(out_path)

    def test_jsonl_epub_only(self, tmp_path, config):
        src = _write_jsonl(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        status, out_path = process_file(src, None, None, config, out_dir, output_format="epub")
        assert status == "ok"
        assert out_path.endswith(".epub") and os.path.exists(out_path)
        assert not os.path.exists(out_path[:-5] + ".md")   # no stray .md
        assert zipfile.is_zipfile(out_path)
        # Body must contain the transcript, NOT the YAML frontmatter
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(out_path) as zf:
            body = zf.read("OEBPS/text/content.xhtml").decode("utf-8")
        ET.fromstring(body)
        assert "測試訊息內容" in body
        assert "pubDate:" not in body and "sourcePath:" not in body


class TestFormatSelection:
    def test_epub_only_no_md(self, tmp_path, config):
        src = _write_txt(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        status, out_path = process_file(src, None, None, config, out_dir, output_format="epub")
        assert status == "ok"
        assert out_path.endswith(".epub") and os.path.exists(out_path)
        assert not os.path.exists(os.path.join(out_dir, "doc.md"))

    def test_both_writes_both(self, tmp_path, config):
        src = _write_txt(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        status, out_path = process_file(src, None, None, config, out_dir, output_format="both")
        assert status == "ok"
        assert out_path.endswith(".md")
        assert os.path.exists(os.path.join(out_dir, "doc.md"))
        assert os.path.exists(os.path.join(out_dir, "doc.epub"))

    def test_epub_write_failure_reports_write_error(self, tmp_path, config, monkeypatch):
        # Simulate the epub write failing after the md was written (both mode)
        src = _write_txt(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        real_replace = os.replace

        def failing_replace(srcp, dstp):
            if str(dstp).endswith(".epub"):
                raise OSError("disk full")
            return real_replace(srcp, dstp)
        monkeypatch.setattr("cleaner.os.replace", failing_replace)
        status, out_path = process_file(src, None, None, config, out_dir, output_format="both")
        assert status == "write_error"
        assert out_path is None
        # the successfully written md is kept on disk (spec-pinned behavior)
        assert os.path.exists(os.path.join(out_dir, "doc.md"))


class TestRunOnePreviewField:
    def _convert(self, tmp_path, fmt):
        from core import convert_file
        src = _write_txt(tmp_path, name=f"doc_{fmt}.txt")
        out_dir = str(tmp_path / f"out_{fmt}")
        os.makedirs(out_dir)
        return convert_file(src, output_dir=out_dir, output_format=fmt)

    def test_md_has_preview(self, tmp_path):
        result = self._convert(tmp_path, "md")
        assert result["status"] == "ok"
        assert result["preview"] == result["output"]
        assert result["preview"].endswith(".md")

    def test_both_preview_is_md(self, tmp_path):
        result = self._convert(tmp_path, "both")
        assert result["preview"].endswith(".md")

    def test_epub_only_no_preview(self, tmp_path):
        result = self._convert(tmp_path, "epub")
        assert result["status"] == "ok"
        assert result["output"].endswith(".epub")
        assert result["preview"] is None

    def test_error_no_preview(self, tmp_path):
        from unittest.mock import patch
        from core import convert_file
        src = _write_txt(tmp_path)
        with patch("core.process_file", return_value=("write_error", None)):
            result = convert_file(src, output_dir=str(tmp_path), output_format="both")
        assert result["preview"] is None


class TestCollisionAndMdFailure:
    def test_both_mode_collision_suffix_shared_stem(self, tmp_path, config):
        src = _write_txt(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        # Pre-existing doc.md forces the _1 suffix for BOTH outputs
        with open(os.path.join(out_dir, "doc.md"), "w") as f:
            f.write("existing")
        status, out_path = process_file(src, None, None, config, out_dir, output_format="both")
        assert status == "ok"
        assert out_path.endswith("doc_1.md")
        assert os.path.exists(os.path.join(out_dir, "doc_1.md"))
        assert os.path.exists(os.path.join(out_dir, "doc_1.epub"))

    def test_md_write_failure_reports_write_error(self, tmp_path, config, monkeypatch):
        src = _write_txt(tmp_path)
        out_dir = str(tmp_path / "out")
        os.makedirs(out_dir)
        real_replace = os.replace

        def failing_replace(srcp, dstp):
            if str(dstp).endswith(".md"):
                raise OSError("disk full")
            return real_replace(srcp, dstp)
        monkeypatch.setattr("cleaner.os.replace", failing_replace)
        status, out_path = process_file(src, None, None, config, out_dir, output_format="md")
        assert status == "write_error"
        assert out_path is None
