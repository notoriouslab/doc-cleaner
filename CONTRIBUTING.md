# Contributing

Contributions are welcome — bug reports, prompt templates, ad truncation patterns, and pull requests.

貢獻方式：bug 回報、提示詞範本、廣告截斷正則、Pull Request。

## Adding Ad Truncation Patterns / 新增廣告截斷正則

The easiest contribution — add regex patterns for your bank/institution's boilerplate footers:

最簡單的貢獻方式 — 新增你銀行或機構的固定廣告尾巴正則：

1. Fork the repo and create a branch: `git checkout -b add-<institution>-pattern`
2. Add the pattern to `config.example.json` under `ad_truncation_patterns`
3. Test with a sample document: `python cleaner.py --input sample.pdf --ai none --verbose`
4. Open a PR with the subject: `pattern: add <Institution Name>`

## Adding Prompt Templates / 新增提示詞範本

If you work with a specific document type (medical, legal, academic, etc.):

1. Create a new file in `prompts/` (e.g. `prompts/medical.txt`)
2. Follow the JSON output schema: `title`, `summary`, `refined_markdown`, `tags`
3. Test with: `python cleaner.py --input sample.pdf --ai gemini --verbose`
4. Open a PR with the subject: `prompt: add <document-type> template`

## Bug Reports / 回報問題

Please include:
- Python version (`python --version`)
- AI backend used (`gemini`, `groq`, `ollama`, or `none`)
- File type that triggered the issue (PDF, DOCX, XLSX, etc.)
- Anonymised log output: `python cleaner.py --input <file> --verbose 2>&1`
- What you expected vs. what happened

## Pull Requests

- Keep PRs focused — one fix or feature per PR
- All `.py` files must pass `python -m py_compile <file>` with no errors
- Do not commit `.env` or `config.json`
- Update `README.md` if you add CLI flags or config fields
- Preserve existing table formatting in output — tables are first-class citizens here

## Development Setup / 開發環境

```bash
git clone https://github.com/notoriouslab/doc-cleaner.git
cd doc-cleaner

# Core dependencies
pip install PyMuPDF python-docx pandas openpyxl Pillow

# (Optional) AI backend
pip install google-genai python-dotenv   # Gemini
# Groq uses its OpenAI-compatible API directly; set GROQ_API_KEY in .env or your shell
pip install ollama                        # Ollama

# Config
cp config.example.json config.json
cp .env.example .env
# Edit .env — set GEMINI_API_KEY or GROQ_API_KEY if using a cloud backend

# Test: raw extraction (no AI)
python cleaner.py --input sample.pdf --ai none --verbose

# Test: with AI
python cleaner.py --input sample.pdf --ai gemini --verbose
```
