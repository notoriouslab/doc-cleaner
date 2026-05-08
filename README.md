# doc-cleaner

[![GitHub release](https://img.shields.io/github/v/release/notoriouslab/doc-cleaner)](https://github.com/notoriouslab/doc-cleaner/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)

**結構化文件轉 Markdown —— 支援 PDF、DOCX、XLSX、PPTX、PPT、DOC、DXF、純文字。中文友好、表格保留、隱私優先。**

屬於 [notoriouslab](https://github.com/notoriouslab) 開源工具組的一員 · 需要 Python 3.9+

> [English README](README.en.md)

---

## 核心定位

市面上大多文件轉 Markdown 工具不是丟掉表格，就是搞壞中文字元，或得把機密文件上傳到雲端。**doc-cleaner 從第一天就為繁體中文設計**，並保留表格完整。

**典型使用：**
- 📊 **金融對帳單** — Big5/CP950 自動偵測，提取交易清單和數字完整無損
- 📄 **多格式批處理** — PDF/DOCX/XLSX/PPTX 混合輸入，統一輸出 Markdown
- 🔒 **隱私優先** — 選用 Ollama 本地推理，文件不上雲端
- 🤖 **AI Agent 整合** — OpenClaw 等框架可直接 shell 呼叫，附帶 `SKILL.md` 支援

### 三大特色

| 特色 | 做法 |
|------|------|
| **表格保留** | DOCX/XLSX → Markdown pipe table；PDF 表格用 opendataloader-pdf 直接提取，不靠 AI 重建 |
| **多格式支援** | PDF、DOCX、XLSX、PPTX、PPT、DOC、DXF、TXT、MD —— 一個工具全搞定 |
| **隱私 & 無 AI 模式** | `--ai none` 純文字提取（零 API key）；或用 Ollama 本地推理 |

---

## 快速開始（三步）

```bash
# 1. Clone
git clone https://github.com/notoriouslab/doc-cleaner.git && cd doc-cleaner

# 2. 安裝
pip install -r requirements.txt

# 3. 執行
python cleaner.py --input ./documents/ --ai none
```

**輸出：** `./output/` 下每個檔案對應的 `.md` 檔案

### 常見使用場景

**場景 1：純文字提取（無 API key）**
```bash
# 最簡單的方式，零成本
python cleaner.py --input statement.pdf --ai none
```

**場景 2：用 Gemini 提高品質（雲端推薦）**
```bash
cp .env.example .env
# 編輯 .env，填入 GEMINI_API_KEY
python cleaner.py --input statement.pdf --ai gemini
```

**場景 3：本地 Ollama（隱私優先）**
```bash
# 需先安裝並啟動 Ollama（見下方 Ollama 選型）
python cleaner.py --input statement.pdf --ai ollama
```

**場景 4：預覽不寫入**
```bash
python cleaner.py --input ./documents/ --dry-run --verbose
```

### 進階安裝（選裝）

高品質 PDF 表格提取、PDF 解密、PPTX/DXF 支援等：

```bash
# 高品質 PDF 提取（推薦）
pip install opendataloader-pdf            # 需要 Java 11+

# PDF 視覺模式（掃描 PDF）
pip install pdf2image                     # 另需 poppler：brew install poppler

# PDF 解密
pip install pikepdf

# 額外格式（PPTX / DXF）
pip install python-pptx ezdxf
```

設定 API key（若使用雲端）：
```bash
cp config.example.json config.json
cp .env.example .env
# 編輯 .env 填入 GEMINI_API_KEY 或 GROQ_API_KEY
```

---

## 核心概念

### PDF 智慧分流

不是所有 PDF 都一樣。doc-cleaner 自動分類後決定處理策略：

| 類型 | 特徵 | 處理方式 |
|------|------|---------|
| **原生文字** | 字元密度 ≥8，亂碼 <5%，短行 ≤70% | 直接提取（快速、免費） |
| **格式破碎** | 短行 >70%（表格被壓扁） | opendataloader-pdf 表格提取 / AI 視覺 + 文字 |
| **掃描圖片** | 字元密度 <8 | PDF 轉圖 + AI 視覺處理 |

**推薦做法（最省錢）：**
```bash
# 步驟 1：全部用 --ai none 提取（快速、免費、隱私）
python cleaner.py --input ./documents/ --ai none --output-dir ./output/raw

# 步驟 2：檢查 log，只對「掃描圖片」的檔案跑 AI
python cleaner.py --input scanned.pdf --ai gemini
```

### 廣告清洗

台灣金融業對帳單常見投資風險告知、法律聲明等固定內容。兩種清洗機制：

| 機制 | 行為 | 場景 |
|------|------|------|
| **尾部截斷** | 第一次匹配後全部截掉 | 文件尾部的法律聲明 |
| **中間移除** | 單獨移除該段落 | 夾在中間的行銷廣告 |

在 `config.json` 設定：
```json
{
  "ad_truncation_patterns": ["謹慎理財.{0,20}信用至上"],
  "ad_strip_patterns": ["※運動賺回饋"]
}
```

安全機制：若截斷會移除 >70% 內容，程式自動跳過並警告。所有正則在啟動時驗證。

### 表格保留

表格在 doc-cleaner 是一等公民：

- **DOCX**：`python-docx` 直接提取 → Markdown pipe table
- **XLSX/CSV**：`pandas.to_markdown()` — 所有工作表
- **PDF**：opendataloader-pdf 直接輸出完整 pipe table（無需 AI）
- **AI 提示詞**：明確指示保留現有表格原樣

### 隱私和安全

| 選項 | 效果 |
|------|------|
| `--ai none` | 零 API key、零雲端，本機純提取 |
| `--ai ollama` | 本地 Ollama 推理，文件不上網 |
| `--ai gemini` / `--ai groq` | 雲端推理，更高品質 |

其他安全機制：
- **原子寫入** — 臨時檔 + `os.replace()`，無半殘輸出
- **機密隔離** — API key 只在 `.env`，啟動時自動檢查
- **OOM 防護** — PDF 視覺模式預設最多 15 頁（可調整）
- **JSON 降級** — AI 回傳失效時自動降級為 raw text

---

## 進階參考

### 命令列選項

```
python cleaner.py [選項]

  --input, -i       要處理的檔案或目錄（必填，不遞迴）
  --output-dir, -o  輸出目錄（預設：./output）
  --config          設定檔路徑（預設：<程式目錄>/config.json）
  --ai              gemini | groq | ollama | none（預設：config 或 gemini）
  --password        PDF 解密密碼（優先於 .env 和 config）
  --summary         輸出 JSON 摘要到 stdout（供腳本/Agent 解析）
  --dry-run         預覽不寫入
  --verbose         除錯日誌
  --version         版本資訊
```

**Exit code：** `0` = 全部成功 · `1` = 部分失敗 · `2` = 設定錯誤

### 設定檔 (config.json)

```jsonc
{
  "ai": {
    "backend": "gemini",                      // 預設後端
    "prompt_template": "prompts/default.txt", // 提示詞路徑
    "gemini": { "model": "gemini-2.5-pro" },
    "groq": {
      "model": "meta-llama/llama-4-scout-17b-16e-instruct",
      "timeout": 120
    },
    "ollama": {
      "model": "qwen3.5:9b",
      "host": "http://localhost:11434"
    }
  },
  "pdf": {
    "dpi": 200,
    "max_pages": 15
  },
  "output": { "frontmatter": true },
  "ad_truncation_patterns": ["謹慎理財.{0,20}信用至上"],
  "ad_strip_patterns": ["※運動賺回饋"]
}
```

**機密管理：** API key 只在 `.env`，**不可**放 `config.json`。啟動時自動驗證。

```bash
# .env 範例
GEMINI_API_KEY=...
GROQ_API_KEY=...
PDF_PASSWORD=...
```

### 自訂 AI 提示詞

doc-cleaner 內建 2 個提示詞範本：

| 檔案 | 用途 |
|------|------|
| `prompts/default.txt` | 通用文件清洗 |
| `prompts/finance.txt` | 銀行對帳單、財務報表 |

**自訂：** 在 `prompts/` 新增 `.txt` 檔，AI 輸出必須是 JSON：

```json
{
  "title": "簡短標題",
  "summary": "1-2 句摘要",
  "refined_markdown": "完整清洗後 Markdown",
  "tags": ["標籤1", "標籤2"]
}
```

### Ollama 選型

表格重建是高難度，小模型力不從心。資源夠可試試 qwen3.5 系列（原生視覺）：

| 模型 | 大小 | 視覺 | 表格重建 | 中文 | 建議 |
|------|------|------|---------|------|------|
| `qwen3.5:27b` | 17 GB | ✓ | 好 | 優 | 效果最佳 |
| `qwen3.5:9b` | 6.6 GB | ✓ | 可 | 好 | **預設**，平衡最佳 |
| `qwen3.5:4b` | 3.4 GB | ✓ | 差 | 可 | 輕量但表格勉強 |
| `qwen3:30b` | 19 GB | — | 好 | 優 | MoE 快速，無視覺 |

**建議：** `qwen3.5:9b` 可跑掃描 PDF；`qwen3:30b` 只用原生文字 PDF。8GB RAM 用戶建議用 `--ai gemini` 或 `--ai none`。

### 支援格式完整表

| 格式 | Parser | 表格 | 備註 |
|------|--------|------|------|
| **PDF（原生）** | opendataloader-pdf / PyMuPDF | pipe table / AI 重建 | ODL 優先 |
| **PDF（掃描）** | pdf2image → AI 視覺 | AI 重建 | 需 poppler |
| **PDF（加密）** | pikepdf | pipe table / AI 重建 | 選裝 |
| **DOCX** | python-docx | pipe table | 跨平台 |
| **XLSX / XLS** | pandas | pipe table | 全工作表 |
| **CSV** | pandas | pipe table | 自動偵測 |
| **PPTX** | python-pptx | pipe table | 投影片+備忘錄 |
| **PPT** | macOS textutil | — | 純文字，macOS only |
| **DOC** | macOS textutil | — | 純文字，macOS only |
| **DXF** | ezdxf | — | 工程圖文字、尺寸 |
| **TXT / MD** | stdlib | — | Big5/CP950/UTF-16 |

---

## 整合與生態

### AI Agent 框架

doc-cleaner 是標準 CLI，任何 AI agent 框架可透過 shell 呼叫。附帶 `SKILL.md` 供 [OpenClaw](https://openclaw.ai/) 使用。

```bash
# Agent 範例：處理 + JSON 摘要
python cleaner.py --input document.pdf --ai none --summary
```

`--summary` 輸出：
```json
{"version":"1.0.0","total":1,"success":1,"failed":0,"files":[{"file":"document.pdf","output":"./output/document.md","status":"ok"}]}
```

### notoriouslab 組合拳

```
gmail-statement-fetcher  →  Gmail 自動下載 PDF 對帳單
          ↓
    doc-cleaner          →  PDF/DOCX/XLSX → 結構化 Markdown
          ↓
   personal-cfo          →  月度審計 + 退休滑翔路徑（開發中）
```

各工具獨立可用，合併使用構成完整個人財務自動化流水線。

---

## 安全政策詳見 [SECURITY.md](SECURITY.md)

---

## 貢獻

最簡單的貢獻方式：

1. **新增廣告正則** — 加入你銀行的截斷/移除規則到 `config.example.json`
2. **新增提示詞範本** — 在 `prompts/` 建立新的 `.txt` 檔
3. **回報編碼問題** — 附上匿名化樣本和 log

詳見 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=notoriouslab/doc-cleaner&type=Date)](https://star-history.com/#notoriouslab/doc-cleaner&Date)

---

## 授權

MIT
