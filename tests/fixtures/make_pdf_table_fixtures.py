"""Generate PDF table fixtures for tests/test_pdf_tables.py.

Run manually (requires reportlab, which is NOT a project dependency):

    python3 tests/fixtures/make_pdf_table_fixtures.py

The generated PDFs are committed alongside this script so the test suite
only needs PyMuPDF to read them. Fixtures mirror the real-world defect
cases that motivated the fix-pdf-table-extraction change:

- merged_cells.pdf          row-span title + full-width column-span row
- crosspage_repeated_header.pdf  60-row table across 2 pages, repeated header
- special_chars.pdf         cells containing ``|`` and newlines
- no_header.pdf             plain data table with no styled header row
- mixed_content.pdf         paragraph above and below a table
"""
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
_styles = getSampleStyleSheet()
for _style in _styles.byName.values():
    _style.fontName = "STSong-Light"

GRID = ("GRID", (0, 0), (-1, -1), 0.5, colors.black)
CJK_FONT = ("FONTNAME", (0, 0), (-1, -1), "STSong-Light")
HEADER_BG = ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)


def _p(text):
    return Paragraph(text, _styles["Normal"])


def _build(filename, story):
    path = os.path.join(FIXTURE_DIR, filename)
    SimpleDocTemplate(path, pagesize=A4, invariant=1).build(story)
    print(f"wrote {path}")


def merged_cells():
    data = [
        [_p("項目"), _p("年度"), _p("金額"), _p("備註")],
        [_p("經常性支出小計說明"), "", "", ""],
        [_p("長期趨勢與成效評估說明（本欄跨三行呈現）"), _p("2023"), _p("100"), _p("初期")],
        ["", _p("2024"), _p("120"), _p("成長")],
        ["", _p("2025"), _p("135"), _p("穩定")],
        [_p("單項支出"), _p("2025"), _p("42"), _p("一次性")],
    ]
    table = Table(data, colWidths=[150, 70, 70, 120])
    table.setStyle(TableStyle([
        GRID, HEADER_BG,
        ("SPAN", (0, 1), (-1, 1)),   # column-span section row
        ("SPAN", (0, 2), (0, 4)),    # row-span title cell
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    _build("merged_cells.pdf", [_p("壹、基本資料與成效評估"), Spacer(1, 12), table])


def crosspage_repeated_header():
    data = [[_p("編號"), _p("金額")]]
    for i in range(1, 61):
        data.append([_p(f"{i:03d}"), _p(str(i * 10))])
    table = Table(data, colWidths=[100, 100], repeatRows=1)
    table.setStyle(TableStyle([GRID, HEADER_BG]))
    _build("crosspage_repeated_header.pdf", [table])


def special_chars():
    data = [
        [_p("欄位"), _p("內容")],
        [_p("含管線 a|b 的值"), _p("第一行<br/>第二行")],
        [_p("一般值"), _p("純文字")],
    ]
    table = Table(data, colWidths=[150, 200])
    table.setStyle(TableStyle([GRID, HEADER_BG]))
    _build("special_chars.pdf", [table])


def no_header():
    data = [
        [_p("甲類"), _p("100")],
        [_p("乙類"), _p("200")],
        [_p("丙類"), _p("300")],
    ]
    table = Table(data, colWidths=[120, 120])
    table.setStyle(TableStyle([GRID]))
    _build("no_header.pdf", [table])


def mixed_content():
    data = [
        [_p("科目"), _p("金額")],
        [_p("奉獻收入"), _p("500")],
    ]
    table = Table(data, colWidths=[120, 120])
    table.setStyle(TableStyle([GRID, HEADER_BG]))
    _build("mixed_content.pdf", [
        _p("表格上方的說明段落。"), Spacer(1, 12),
        table, Spacer(1, 12),
        _p("表格下方的結論段落。"),
    ])


def header_grid_only():
    """Header row fully gridlined; body rows have only horizontal rules —
    reproduces the statement layout whose body rows collapse into one cell
    (proven recipe: BOX + header GRID + LINEBELOW)."""
    data = [
        ["類別", "代號", "名稱", "數量", "金額"],
        ["現股", "2330", "台積電", "1,000", "580,000"],
        ["現股", "0050", "元大台灣50", "2,000", "270,000"],
        ["小計標題列跨全寬置中", "", "", "", ""],
    ]
    table = Table(data, colWidths=[70, 70, 120, 80, 90])
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ("GRID", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.black),
        CJK_FONT,
        ("SPAN", (0, 3), (-1, 3)),
        ("ALIGN", (0, 3), (-1, 3), "CENTER"),
    ]))
    _build("header_grid_only.pdf", [table])


if __name__ == "__main__":
    merged_cells()
    crosspage_repeated_header()
    special_chars()
    no_header()
    mixed_content()
    header_grid_only()
