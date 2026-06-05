Doc Cleaner v1.4.0
==================

文件清洗工具——將各種文件轉換為 Markdown 格式。
純文字提取，無需 AI，無需網路連線，無需設定。


【使用方法】

1. 開啟 App
2. 點「選擇檔案」按鈕，或將文件拖放到視窗中
3. 選擇輸出位置：
   • 同資料夾 — 輸出的 .md 放在原始檔案旁邊
   • 桌面 — 所有輸出集中放在桌面
4. 按「轉換」
5. 完成後點「在 Finder 顯示」查看輸出檔案


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
• TXT / MD — 純文字直接輸出
• JSONL    — Claude Code session transcript，對話紀錄轉結構化 Markdown
             （此格式為 Claude Code 專用，不適用於一般 JSONL 檔案）


【首次開啟 App】

此 App 尚未通過 Apple 公證，首次開啟時 macOS 會顯示安全性警告。
以下依 macOS 版本說明解決方式：

▸ macOS 13 Ventura 及以前
  在 App 圖示上按右鍵（或 Control+點一下）→ 選「開啟」
  在跳出的對話框中再按一次「開啟」即可

▸ macOS 14 Sonoma / macOS 15 Sequoia（推薦）
  系統設定 → 隱私權與安全性 → 往下滑
  找到「Doc Cleaner」→ 點「仍要開啟」

▸ 或使用 Terminal（適合熟悉指令列的使用者）
  將 App 拖到 Applications 後，在 Terminal 執行一次：
  xattr -cr /Applications/Doc\ Cleaner.app

確認一次後，之後每次開啟都正常，不會再詢問。


【注意事項】

• 掃描版 PDF（整頁為圖片）無法提取文字，會顯示 ❌
• DOC / PPT 使用 macOS 系統內建工具轉換，通常在 60 秒內完成，無需安裝任何軟體
• 若同名輸出檔已存在，自動加 _1、_2 後綴避免覆蓋


notoriouslab © 2026
https://github.com/notoriouslab/doc-cleaner
