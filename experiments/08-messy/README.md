# 08 把乾淨法規故意弄亂：哪種亂弄壞 pipeline 的哪一步、掉幾分、修它要什麼

## 1. 想回答的問題

01–07 都在同一份乾淨的法規 PDF 上做（有文字層、單欄、表格有框線）。真資料「可能很亂」但還沒拿到。把這份法規一次弄亂一種——頁首頁尾、兩欄、跟另一份法規串接、手機拍、影印三代、表格沒框線——走原本的 pipeline，檢索與讀表各掉幾分、掉在哪一步？目標是一張對照表，真資料來了看它屬於哪幾列就知道先修哪裡。

## 2. 怎麼做

**別人怎麼解的**：
- OHRBench「OCR Hinders RAG」(arXiv 2412.02592)：把 OCR 噪音分**語意噪音**（認錯字）與**格式噪音**（粗體、換行、表格線的表示法），語意噪音用八種影像退化（背景、椒鹽、旋轉、二值化、扭曲、陰影、模糊…）做出三級再跑 OCR；七個領域含 Law。最好的 OCR 跟正解仍差 14%，重度退化檢索與生成掉近 50%，表格題掉得比純文字多。本實驗的掃描兩級照它的操作清單挑五種。
- StruNRAG (ACL 2026 Findings)：三種**結構噪音**（行插入、段落交錯、行交錯），結論是結構噪音穩定傷檢索、生成反而耐。本實驗的兩欄交錯就是它的「行交錯」，所以文字軌只量檢索。
- Augraphy (ICDAR 2023)：印表／掃描退化的標準庫，依賴太重（opencv、numba、scikit）沒裝，用 Pillow 做同類操作。
- 頁首頁尾：unstructured、rag-scrubber 都靠「跨頁重複的行 + 位置」偵測；**沒找到**量過「頁首頁尾進 chunk 讓檢索掉多少」的研究，只有部落格宣稱 15–40%。無框線表：Camelot stream、pdfplumber `text` 策略是標準替代，沒有繁中數字。兩欄：pdfplumber 按 y 排序會交錯、pypdf 照串流序——社群常識、沒人量。

**弄亂一律在 PDF 層做**（`degrade.py`，PyMuPDF 內建 CJK 字型；PDF／PNG 在 `data/08-messy/`，不進 git），之後走各軌原本的 pipeline，不給任何提示。

**文字軌**（`run_text.py`，01 的 30 題 + 10 題保留 = 40 條文題，dense bge-m3 ／ bm25 ／ random，hit@1000）：

| 變體 | 怎麼做 | 抽字 | 頁數／字元 |
|---|---|---|---|
| clean | 01 的 `law.pdf` | pypdf | 283／255k |
| header-footer | 每頁蓋「用戶用電設備裝置規則」左上、修正日期右上、「第 N 頁／共 283 頁」右下 | pypdf | 283／269k |
| two-column | 文字層的 9,383 行重排成 A4 兩欄（10 pt，字元折行成 14,669 行） | pypdf **和** pdfplumber | 157／260k |
| concat | `law.pdf` + 輸配電設備裝置規則（J0030019，從全國法規資料庫頁面取文、排成單欄 47 頁、286 條） | pypdf | 330／321k |

切法：article（01 的一條一塊）與 fixed-1000（解析壞掉時的保底），01 的 109 張表塊照 07 一律加進去。**標籤不靠解析**（`labels.py`）：把乾淨文字每一行記上它屬於哪一條，只留「≥ 8 字、只屬於一條」的行當見證（「依下列規定辦理：」這種樣板行、`第 N 條` 標題行都不算，後者因為第二份法規也有）；chunk 裡整行出現某條的見證行就標那條（行邊界允許空格，因為 pdfplumber 把兩欄用空格黏成一行）。在乾淨文字上這套標籤跟 01 的解析 1050 塊全部一致（測試）。另報每個變體 `parse_articles` 找到幾條。

**表格軌**（`run_table.py`、`score.py`、`report.py`：06 的程式；03 的 82 題逐字；markdown 餵法；qwen3:8b `think:false`、gpt-oss:20b `think:"low"`、temperature 0、seed 0）：

| 條件 | 怎麼做 | 抽表 |
|---|---|---|
| closed-book | 不餵 | 雜訊底線 |
| clean | 04 的 markdown | pdfplumber 幾何（盲測基線，同輪重跑） |
| scan-clean | 06 的 300 dpi 頁圖 | glm-ocr（提示尾換行）→ HTML → markdown（06 的 glm2 輸出照用，答題重跑） |
| scan-phone | 同頁圖：旋轉 1.5°、高斯模糊 σ1、左亮右暗 25% 的陰影、JPEG q60、降到 200 dpi | 同上，重跑 OCR |
| scan-copy3 | 再加椒鹽 1%、閾值 160 二值化、降到 150 dpi | 同上，重跑 OCR |
| borderless-text | 04 `tables.json` 的格子重排成**沒有任何線**的 PDF（欄對齊、標題、註照舊、字縮到放得下） | pdfplumber `text` 策略 → 03 管線（header 1 列）→ markdown |
| borderless-flat | 同一份 PDF | lines 策略找不到表時的真實路徑：`table_to_text` 攤平 |

結構層（`structure.py`，06 的）：每條路線對 04 的格子正解算格子／數字／列／文字召回。

**資料體檢**（`doc_health.py`）：丟一個資料夾，每個 PDF 一列：頁數、有文字層的頁、lines 策略找到表的頁、「沒框線但 ≥ 5 列對齊」的頁、兩欄的頁（字的框分左右兩群、幾乎沒有跨中線）、頁緣重複的行（數字摺成 #、≥ 50% 的頁出現、不在表內）。08 的變體就是它的測試案例。

**五條規矩**：雜訊底線 = random／closed-book；盲測基線 = clean（兩軌同輪重跑）；區間 = bootstrap 95%、每個變體對 clean 配對差距；記版本 = 下表；漂移探針 = 向量 cache 對新算 cosine、答題每格重打 2 題。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-29／mini.local |
| ollama | 0.40.2 |
| qwen3:8b digest | `sha256:500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` |
| gpt-oss:20b digest | `sha256:f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe` |
| glm-ocr | `89d735dd9c07`（0.89B F16） |
| bge-m3 | 同 01（`results/text/meta.json`） |
| Python／pypdf／pdfplumber／PyMuPDF | 3.12.15／6.21.3／0.11.10／2.0（MuPDF 1.28.5） |

## 3. 數字

（跑完填）

## 4. 結論

（跑完填）

## 5. 沒做到的、下次想試的

（跑完填）
