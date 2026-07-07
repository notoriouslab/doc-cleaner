Doc Cleaner v1.6.1
==================

文件清洗工具——將各種文件轉換為 Markdown 格式。
純文字提取，無需 AI，無需網路連線，無需設定。


【使用方法】

1. 開啟 App
2. 點「選擇檔案」按鈕，或將文件拖放到視窗中
   （也可直接拖放整個資料夾，會自動轉換裡面所有支援的檔案，含子資料夾）
3. 選擇輸出位置：
   • 同資料夾 — 輸出的 .md 放在原始檔案旁邊
   • 桌面 — 所有輸出集中放在桌面
   • 選擇資料夾… — 自選任一資料夾（會記住，下次沿用）
4. 按「轉換」
5. 完成後可點「預覽」直接在 App 內檢視轉好的 Markdown，
   或點「在 Finder 顯示」查看輸出檔案


【支援格式】

• PDF      — 僅原生文字 PDF（掃描版圖片 PDF 無法提取）
• DOCX     — Word 文件，表格結構完整保留
• DOC      — 舊版 Word（透過 macOS 系統內建工具轉換，無需安裝）
• XLSX     — Excel 試算表，每個工作表分節輸出，表格格式保留
• XLS      — 舊版 Excel
• CSV      — 逗號分隔值
• PPTX     — PowerPoint，含備忘錄
• PPT      — 舊版 PowerPoint（透過 macOS 系統內建工具轉換，無需安裝）
• DXF      — AutoCAD 工程圖，提取文字標註與尺寸
• NUMBERS  — Apple Numbers 試算表，每個表格分節輸出
• KEY      — Apple Keynote 簡報，每張投影片分節（依閱讀順序）
• PAGES    — Apple Pages 文件（見下方注意事項）
• EPUB     — 電子書，依章節分節，含書名與作者
• JSONL    — Claude Code 對話記錄
• TXT / MD — 純文字直接輸出


【首次開啟 App】

此 App 尚未通過 Apple 公證，首次開啟時 macOS 會顯示安全性警告。
以下依 macOS 版本說明解決方式：

▸ macOS 13 Ventura 及以前
  在 App 圖示上按右鍵（或 Control+點一下）→ 選「開啟」
  在跳出的對話框中再按一次「開啟」即可

▸ macOS 14 Sonoma / macOS 15 Sequoia（推薦）
  系統設定 → 隱私權與安全性 → 往下滑
  找到「Doc Cleaner」→ 點「仍要開啟」

確認一次後，之後每次開啟都正常，不會再詢問。


【注意事項】

• 掃描版 PDF（整頁為圖片）無法提取文字，會顯示 ❌
• DOC / PPT 使用 macOS 系統內建工具轉換，通常在 60 秒內完成，無需安裝任何軟體
• Pages：新版 Pages（2020 後）的內容無法直接提取，請在 Pages 中
  「檔案 → 輸出 → PDF」後再轉換該 PDF；舊版含內嵌預覽的檔案則可直接轉換
• 受 DRM 保護的 EPUB 無法提取
• 若同名輸出檔已存在，自動加 _1、_2 後綴避免覆蓋


====================================================================


Doc Cleaner v1.6.1
==================

A document-cleaning tool — converts various documents to Markdown.
Plain-text extraction. No AI, no network, no configuration.


[How to use]

1. Open the app
2. Click "Choose Files", or drag documents into the window
   (you can also drop a whole folder — every supported file inside it,
   including subfolders, is converted)
3. Pick an output location:
   • Same folder — the .md output sits next to the source file
   • Desktop — all output collected on the Desktop
   • Choose folder… — pick any folder (remembered for next time)
4. Click "Convert"
5. When done, click "Preview" to view the converted Markdown in-app,
   or "Show in Finder" to view the output file


[Supported formats]

• PDF      — text-native PDFs only (scanned image-only PDFs can't be extracted)
• DOCX     — Word documents, table structure preserved
• DOC      — legacy Word (converted via a built-in macOS tool, no install needed)
• XLSX     — Excel spreadsheets, one section per sheet, tables preserved
• XLS      — legacy Excel
• CSV      — comma-separated values
• PPTX     — PowerPoint, including speaker notes
• PPT      — legacy PowerPoint (converted via a built-in macOS tool)
• DXF      — AutoCAD drawings, extracts text labels and dimensions
• NUMBERS  — Apple Numbers spreadsheets, one section per table
• KEY      — Apple Keynote, one section per slide (in reading order)
• PAGES    — Apple Pages documents (see Notes below)
• EPUB     — e-books, one section per chapter, with title and author
• JSONL    — Claude Code conversation transcripts
• TXT / MD — plain text, passed through


[First launch]

This app is not yet notarized by Apple, so macOS shows a security warning the
first time you open it. By macOS version:

▸ macOS 13 Ventura and earlier
  Right-click (or Control-click) the app icon → choose "Open"
  Click "Open" again in the dialog

▸ macOS 14 Sonoma / macOS 15 Sequoia (recommended)
  System Settings → Privacy & Security → scroll down
  Find "Doc Cleaner" → click "Open Anyway"

After confirming once, it opens normally every time afterwards.


[Notes]

• Scanned PDFs (whole pages as images) can't yield text and show ❌
• DOC / PPT use a built-in macOS tool, usually finishing within 60 seconds,
  with nothing to install
• Pages: modern Pages files (post-2020) can't be extracted directly — in Pages,
  use File → Export → PDF, then convert that PDF; older files with an embedded
  preview convert directly
• DRM-protected EPUB files can't be extracted
• If an output file of the same name exists, _1 / _2 suffixes are added


notoriouslab © 2026
https://github.com/notoriouslab/doc-cleaner
