# doc-cleaner

將 PDF、DOCX、XLSX 及純文字檔轉換為乾淨的結構化 Markdown。

中文友好、表格友好、隱私優先。

**需要 Python 3.9+** · 屬於 [notoriouslab](https://github.com/notoriouslab) 開源工具組的一員。

> [English README](README.en.md)

---

## 為什麼需要這個工具

市面上大多數文件轉 Markdown 的工具，不是丟掉表格、就是搞壞中文字元，或者得把機密文件上傳到雲端。這也是 AI Agent 整合（OpenClaw 等）， 任何 AI agent 框架都可以透過 shell 呼叫，附帶 `SKILL.md` 讓 [OpenClaw](https://openclaw.ai/) 直接使用。

doc-cleaner 從第一天就為**繁體中文的文件**設計：

| 特色         | 說明                                     |
| ---------- | -------------------------------------- |
| 中文友好       | Big5 / CP950 / UTF-16 自動偵測，金融業對帳單也能用   |
| 表格保留       | DOCX + XLSX → Markdown pipe table，不丟格式 |
| PDF 智慧分流   | 自動辨識：原生文字 / 格式破碎 / 掃描圖片                |
| AI 結構化     | Gemini 雲端、Groq 雲端 或 Ollama 本地             |
| 無 AI 模式（選） | `--ai none` 純提取，零 API、零雲端              |
| PDF 解密（選）  | 選裝 pikepdf                             |
| 廣告截斷       | 可自訂正則，移除副檔內的固定廣告尾巴                     |
| 隱私優先（選）    | Ollama 本地推理，文件不離開你的電腦                  |
| 原子寫入       | 臨時檔 + `os.replace()`，不會產生半殘輸出          |
| 預覽模式       | `--dry-run` 先看再動手                      |

---

## 快速開始

```bash
# 1. 下載
git clone https://github.com/notoriouslab/doc-cleaner.git
cd doc-cleaner

# 2. 安裝核心依賴
pip install -r requirements.txt

# 3.（選裝）AI 後端
pip install google-genai python-dotenv   # Gemini 雲端
# 或
# Groq 雲端不需要額外 SDK；只要設定 GROQ_API_KEY
# 或
pip install ollama                        # Ollama 本地

# 4.（選裝）PDF 擴充
pip install pikepdf                       # PDF 解密
pip install pdf2image                     # PDF 視覺模式（另需安裝 poppler）

# 5. 設定
cp config.example.json config.json        # 編輯 AI 模型、廣告正則等
cp .env.example .env                      # 填入 GEMINI_API_KEY / GROQ_API_KEY（依後端而定）

# 6. 執行
python cleaner.py --input 對帳單.pdf
# 輸出：./output/對帳單.md
```

### 無 AI 模式（最簡單）

不需要任何 API key 或雲端服務，純提取文字和表格：

```bash
pip install -r requirements.txt
python cleaner.py --input ./downloads/ --ai none
```

### 預覽模式

先看會處理哪些檔案，不實際寫入：

```bash
python cleaner.py --input ./downloads/ --dry-run --verbose
```

---

## 命令列選項

```
python cleaner.py [選項]

  --input, -i       要處理的檔案或目錄（必填，不遞迴子目錄）
  --output-dir, -o  輸出目錄（預設：./output）
  --config          設定檔路徑（預設：<程式目錄>/config.json）
  --ai              gemini | groq | ollama | none（預設：config 裡的 backend 或 gemini）
  --password        PDF 解密密碼（優先於 .env 和 config）
  --summary         處理完後輸出 JSON 摘要到 stdout（方便腳本和 AI agent 解析）
  --dry-run         預覽不寫入
  --verbose         啟用除錯日誌
  --version         顯示版本
```

### Exit Code

| Code | 意義 |
|---|---|
| 0 | 全部檔案處理成功 |
| 1 | 部分檔案失敗 |
| 2 | 無可處理的檔案，或設定錯誤 |

---

## 設定檔說明

主設定檔為 `config.json`（從 `config.example.json` 複製），格式如下：

```jsonc
{
  "ai": {
    "backend": "gemini",                        // 預設 AI 後端
    "prompt_template": "prompts/default.txt",   // 提示詞範本路徑
    "gemini": {
      "model": "gemini-2.5-pro"
    },
    "groq": {
      "model": "meta-llama/llama-4-scout-17b-16e-instruct",
      "base_url": "https://api.groq.com/openai/v1",
      "timeout": 120
    },
    "ollama": {
      "model": "qwen3.5:9b",
      "host": "http://localhost:11434"
    }
  },
  "pdf": {
    "dpi": 200,                                 // 視覺模式解析度
    "max_pages": 15                             // 視覺模式最大頁數（防 OOM）
  },
  "output": {
    "frontmatter": true                         // 輸出是否含 YAML 前言
  },
  "ad_truncation_patterns": [                   // 廣告截斷正則（見下文）
    "<投資人權益通知訊息[ >]",
    "謹慎理財.{0,20}信用至上"
  ]
}
```

### 機密管理

- **API Key 和密碼**只能放在 `.env`，**不可**放在 `config.json`
- `config.json` 和 `.env` 都已加入 `.gitignore`，不會被 commit
- 程式啟動時會檢查 `config.json` 是否不小心放了 secret，並發出警告

```
# .env 範例
GEMINI_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
PDF_PASSWORD=your-pdf-password
```

密碼優先順序：`--password` CLI 參數 > `.env` (`PDF_PASSWORD`) > `config.json`

---

## 自訂廣告截斷正則

台灣金融業對帳單 PDF 的尾巴經常帶有投資風險告知、法律聲明等固定文字。doc-cleaner 會在提取文字後、送入 AI 前，把匹配到的位置以後的內容全部截掉。

正則規則放在兩個地方：

| 位置 | 用途 |
|---|---|
| `classifiers/noise.py` 的 `DEFAULT_CUTOFF_PATTERNS` | 內建預設，涵蓋常見台灣金融文件 |
| `config.json` 的 `ad_truncation_patterns` | **使用者自訂**，會覆蓋內建預設 |

**新增你銀行的正則**：直接在 `config.json` 的 `ad_truncation_patterns` 陣列加入即可。

```json
"ad_truncation_patterns": [
  "<投資人權益通知訊息[ >]",
  "[～~]\\s*總\\s*公\\s*司\\s*訊\\s*息\\s*[～~]",
  "依金融監督管理委員會保險局",
  "謹慎理財.{0,20}信用至上",
  "本商品由.{0,30}核准",
  "你的銀行的新正則放這裡"
]
```

安全機制：如果截斷會移除超過 70% 的內容，程式會跳過截斷並警告，防止誤殺。

程式啟動時會預先驗證所有正則語法，有錯會直接報錯退出，不會等到處理檔案才爆。

---

## 自訂 AI 提示詞範本

doc-cleaner 附帶兩個提示詞範本：

| 檔案 | 用途 |
|---|---|
| `prompts/default.txt` | 通用文件清洗 |
| `prompts/finance.txt` | 銀行對帳單、財務報表（保留交易明細、金額） |

在 `config.json` 切換：

```json
"ai": { "prompt_template": "prompts/finance.txt" }
```

**自己寫一個也很簡單**：在 `prompts/` 資料夾新增 `.txt` 檔，AI 輸出必須是 JSON 格式，包含以下欄位：

```json
{
  "title": "簡短標題",
  "summary": "1-2 句摘要",
  "refined_markdown": "完整清洗後的 Markdown 內容",
  "tags": ["標籤1", "標籤2"]
}
```

範例：如果你常處理醫療文件，可以建立 `prompts/medical.txt`，在提示詞裡強調保留病歷編號、日期、診斷代碼等。

---

## PDF 智慧分流

不是所有 PDF 都一樣。doc-cleaner 會先自動分類，再決定處理策略：

| 類型   | 偵測條件                   | 策略           |
| ---- | ---------------------- | ------------ |
| 原生文字 | 字元密度 ≥8，亂碼 <5%，短行 ≤70% | 直接提取（快速、免費）  |
| 格式破碎 | 短行 >70%（表格被壓扁）         | AI 視覺 + 文字兜底 |
| 掃描圖片 | 字元密度 <8                | AI 視覺 + 文字兜底 |

這個分流機制讓原生文字 PDF **不需要浪費 AI 呼叫**，只有真正需要的檔案才送 AI。

### 混合策略（推薦）

最省錢最有效率的做法：

```bash
# 第一步：全部用 raw 模式提取（快速、免費、隱私）
python cleaner.py --input ./downloads/ --ai none --output-dir ./output/raw

# 第二步：檢查 log，只對 "Layout-broken" 或 "Scanned" 的檔案跑 AI
python cleaner.py --input problem_file.pdf --ai gemini --output-dir ./output/ai
# 或
python cleaner.py --input problem_file.pdf --ai groq --output-dir ./output/ai
```

---

## 表格保留

表格在 doc-cleaner 是一等公民：

- **DOCX**：`python-docx` 提取表格 → Markdown pipe table（`|` 分隔符）
- **XLSX/CSV**：`pandas.to_markdown()` — 所有工作表、空格補空字串、每表上限 8000 字元
- **AI 提示詞**：明確指示「保留現有 pipe table 原樣不動」，防止 AI 重排表格

---

## Ollama 模型建議

表格重建是高難度任務，小模型會力不從心。在我的 Macbook Air M2, iMac 2019 上跑這些都不太行，但若你的電腦夠力，可以試試 qwen3.5 全系列支援視覺（Image）：

| 模型            | 大小     | 視覺  | 表格重建 | 中文品質 | 備註                           |
| ------------- | ------ | --- | ---- | ---- | ---------------------------- |
| `qwen3.5:27b` | 17 GB  | 有   | 好    | 優    | **推薦首選** — 原生視覺，256K context |
| `qwen3.5:9b`  | 6.6 GB | 有   | 可    | 好    | **預設值** — 多數機器能跑，掃描 PDF 也行   |
| `qwen3.5:4b`  | 3.4 GB | 有   | 差    | 可    | 純文字可以，表格勉強                   |
| `qwen3:30b`   | 19 GB  | 無   | 好    | 優    | MoE 架構，推理快，但不支援視覺            |


> **建議**：優先選 `qwen3.5` 系列 — 原生視覺意味著掃描 PDF 可以直接送圖片給模型，不需要額外的 OCR。`qwen3.5:27b` 效果最好，`qwen3.5:9b`（6.6GB）是預設值，平衡效果和資源需求。
>
> 如果不需要處理掃描 PDF（只有原生文字 PDF、DOCX、XLSX），`qwen3:30b` 的 MoE 架構推理速度更快。
>
> **8GB 用戶注意**：Ollama 會很慢，建議用 `--ai gemini` 或 `--ai none`。

---

## 支援格式

| 格式             | Parser            | 表格         | 備註                                |
| -------------- | ----------------- | ---------- | --------------------------------- |
| **PDF**（原生文字）  | PyMuPDF (fitz)    | 需 AI 重建    | 文字提取不需 AI，表格結構化才需要                |
| **PDF**（掃描）    | pdf2image → AI 視覺 | 需 AI 重建    | 需安裝 poppler                       |
| **PDF**（加密）    | pikepdf → 上述流程    | 需 AI 重建    | 選裝 pikepdf                        |
| **DOCX**       | python-docx       | pipe table | 直接提取表格；非 macOS 也可用（textutil 僅為兜底） |
| **XLSX / XLS** | pandas + openpyxl | pipe table | 全部工作表                             |
| **CSV**        | pandas            | pipe table | 自動偵測                              |
| **TXT / MD**   | 標準函式庫             | —          | 多編碼支援                             |

### poppler 安裝

PDF 視覺模式（掃描 PDF 轉圖片）需要 poppler 系統套件：

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt-get install poppler-utils
```

不需要視覺模式的話，用 `--ai none` 即可跳過。

---

## 安全性

- **不需雲端**：`--ai ollama` 或 `--ai none` 所有處理都在本機
- **原子寫入**：臨時檔 + `os.replace()`，不會產生半殘輸出
- **機密隔離**：API key 只在 `.env`（不在 `config.json`），啟動時自動檢查
- **OOM 防護**：PDF 視覺模式預設最多 15 頁，可在 config 調整
- **廣告截斷防誤殺**：截斷超過 70% 內容時自動跳過
- **JSON 降級**：AI 回傳的 JSON 解析失敗時，自動降級為 raw text 模式

詳細安全政策請見 [SECURITY.md](SECURITY.md)。

---

## AI Agent 整合（OpenClaw 等）

doc-cleaner 是標準 CLI 工具，任何 AI agent 框架都可以透過 shell 呼叫。附帶 `SKILL.md` 讓 [OpenClaw](https://openclaw.ai/) 直接使用。

```bash
# Agent 呼叫範例：處理檔案 + 取得 JSON 摘要
python cleaner.py --input document.pdf --ai none --summary
```

`--summary` 輸出範例：
```json
{"version":"1.0.0","total":1,"success":1,"failed":0,"files":[{"file":"document.pdf","output":"./output/document.md","status":"ok"}]}
```

Agent 可以用 exit code 判斷成敗（0=全部成功、1=部分失敗、2=設定錯誤），用 `--summary` 的 JSON 取得每個檔案的處理結果。

---

## notoriouslab 組合拳

```
gmail-statement-fetcher   →  從 Gmail 自動下載 PDF 對帳單
        ↓
   doc-cleaner             →  PDF/DOCX/XLSX → 結構化 Markdown
        ↓
   personal-cfo            →  月度審計 + 退休滑翔路徑（開發中）
```

每個工具可獨立使用。合併使用則構成完整的個人財務自動化流水線。

---

## 貢獻

最簡單的貢獻方式：

1. **新增廣告截斷正則** — 加入你銀行的固定尾巴正則到 `config.example.json`
2. **新增提示詞範本** — 在 `prompts/` 建立新的 `.txt` 給不同文件類型
3. **回報編碼問題** — 附上匿名化樣本和 log

詳見 [CONTRIBUTING.md](CONTRIBUTING.md)。

---

## 授權

MIT
