# 06 掃描來源：同一張表從頁面影像 OCR 回來，模型讀表掉多少；三條本地 OCR 路線差多少

## 1. 想回答的問題

01–05 的表都從 PDF 文字層讀，是乾淨的來源。客戶給的常是掃描檔或照片。把 03 的 17 張表轉成 300 dpi 影像，再用三條不花錢、不起服務的本地 OCR 讀回 markdown，餵同一組 82 題：值答對率比乾淨的 markdown 掉多少？三條路線——Apple Vision（macOS 內建）、tesseract chi_tra、glm-ocr（0.9B 視覺模型，ollama）——哪條能用？

## 2. 怎麼做

**別人怎麼解的**（`docs/research/2026-10-29-local-ocr-options.md`、`2026-10-29-parsing-toolchains.md`）：OmniDocBench v1.5 的表格 TEDS 榜上 GLM-OCR 93.9、PaddleOCR-VL 92.8 是本地能跑的最高；通用 VLM（Qwen3-VL-8B）明顯落後專用 OCR 模型；tesseract 的繁中沒有公開 CER，社群回報字間插空白、丟標點；Apple Vision 零依賴但沒有任何繁中或表格的量化評測。**所有榜單都沒有繁體中文分項**，所以要自己量。PaddleOCR 在 Mac 要 paddlepaddle CPU wheel 加 mlx-vlm server，違反這個專案「不起服務」的約定，沒試。

**影像**（`render.py`）：pdfplumber 的 `to_image(resolution=300, antialias=True)`（底層 pypdfium2），18 頁（表五六 2 頁），存 `data/06-ocr/png/`（不進 git，另一台重跑）。這是**乾淨的掃描**：沒有傾斜、沒有雜訊、沒有 JPEG 壓縮，所以這裡的數字是 OCR 路線的**上限**。

**三條路線**（`ocr.py`，輸出進 git：`corpus/<route>/<表號>.txt`）：

| 路線 | 做法 | 17 張表耗時 |
|---|---|---|
| clean | 04 的 markdown（pdfplumber 文字層 + 幾何 span），= 盲測基線，也是結構層的正解 | — |
| vision | `ocrmac`（macOS Vision，`accurate`、`zh-Hant`）回傳行片段與框 → 依 y 中心聚成列（容差 0.6 × 中位高）、列內依 x 排 → 每列一行 markdown，第一列當表頭；「表」開頭的行當標題、「註」開頭的行當註 | 5 秒 |
| tesseract | `tesseract -l chi_tra+eng --psm 6 tsv`，字之間的間距 > 行高就切成新片段，CJK 字間的空白黏回去，之後同 vision 的聚類 | 20 秒 |
| glm | `glm-ocr` 走 host 的 ollama，提示 `Table Recognition:`，回 HTML → 展開 rowspan／colspan 成格子 → markdown | 見第 3 節 |

vision 與 tesseract **不重建欄**：同一列的片段依 x 排，但沒有對齊到欄界，空格子不會留位，所以寬表的列會「左移」。glm 回的是帶 span 的 HTML，有欄。

**結構層**（`structure.py`，不經模型）：以 clean 的格子為正解，算每條路線 (a) 格子召回：正解資料格的文字（NFKC、去空白）有沒有以**一整格**出現；(b) 數字格召回：只看數字格（題目問的就是這些）；(c) 列召回：正解一列的格子有沒有**依序**全出現在 OCR 的某一列裡；(d) 文字召回：格子文字有沒有出現在 OCR 全文任何地方（不管結構）。

**答題層**（`run.py`、`score.py`、`report.py`：03 的程式）：03 的 82 題逐字（測試保證），prompt、schema、模型設定同 02–05，5 條件 × 2 模型 × 82 題 = 820 次。clean 在同一晚重跑當盲測基線（04 看到 gpt-oss 的 md 比 03 漂 3 題）。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-29／mini.local |
| ollama | 0.40.2 |
| qwen3:8b digest | `sha256:500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` |
| gpt-oss:20b digest | `sha256:f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe` |
| glm-ocr | `89d735dd9c07`（glm4 0.89B F16，ollama library） |
| tesseract／chi_tra | 5.5.3／tessdata（legacy+LSTM）2026-10-28 下載 |
| ocrmac | 1.0.x（macOS 26.6.2 Vision） |
| Python／pdfplumber | 3.12.15／0.11.10 |

## 3. 數字

**結構層**（`results/structure.csv`，17 張表平均；clean 全 1.0 是自我檢查）：

| 路線 | 格子召回 | 數字格召回 | 列召回 | 文字召回 | 18 頁耗時 |
|---|---|---|---|---|---|
| glm-ocr | .57 | .59 | .52 | **.97** | 7 分（約 23 秒/頁，M4） |
| vision | **.71** | **.79** | .39 | .73 | 5 秒 |
| tesseract | .15 | .16 | .00 | .22 | 20 秒 |

- **glm-ocr 是兩極的**：9 張小的、有框線的表幾乎全對（表九三～一／二、表二五～六／七、表五五、表五六、表八八～二、表六一 格子召回 ≥ .96，列召回 .9–1.0；表八六～三 多層表頭也對了 .96），**8 張它根本不回表格**——表二五～一／二／四／五（7 欄寬表、資料列沒畫線）、表三六、表九二（長文字格）、表四○（轉置表）只回純文字，字都在（文字召回 .97）但欄列全丟。它回的格式也不固定：markdown、TSV（表二五～六）、純文字三種。
- **Apple Vision 字對、框對，但整欄孤立的數字會被當成直排文字一起丟掉或合成一「行」**（表五五 需量因數 整欄只剩最後一格；表二五～二 安培容量 大半不見），所以列召回只有 .39；文字召回 .73 代表這些數字連文字層都沒有，不是聚類的錯。
- **tesseract chi_tra 不能用**：字間插空白、把表格列讀成「#238 80 ﹣ 200 ˍ 固 ˍ」這種東西，資料格幾乎全丟。


RESULTS_PLACEHOLDER

## 4. 結論

CONCLUSION_PLACEHOLDER

## 5. 沒做到的、下次想試的

- **掃描是乾淨的**：300 dpi、無傾斜、無雜訊。真掃描會更差；下一步加 200 dpi + 0.5° 傾斜 + JPEG q50 的退化版，看哪條路線先垮。
- **vision／tesseract 沒有重建欄**：用所有列的 x 中心聚類出欄界（Doctra 那種 10 px 對齊）再補空格，寬表的列就不會左移。這是 50 行程式，沒做。
- **Apple Vision 會把一整欄孤立的數字當成直排文字**（表五五 的 100/80/70… 整欄不見，`livetext` 框架也一樣，只是把它們合成一個直的「行」）；要救得切成欄條再 OCR，但那要先知道欄在哪。
- **PaddleOCR-VL／PP-StructureV3**：繁中有官方數字（rec 0.747），但 Mac 上要起服務，留到有一整天。
- **沒量 TEDS**：結構層用的是格子召回，不是 TEDS；要跟榜上的數字比得補 TEDS（`teds` 套件或自己寫樹編輯距離）。
- **只有一種 prompt 給 glm-ocr**：`Table Recognition:`；官方還有 `Text Recognition:` 與版面 JSON，沒試。
