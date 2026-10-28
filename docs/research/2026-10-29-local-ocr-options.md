# 本地繁中 OCR 選項（2026-10-29 調查）

> 由 Claude 的子代理用 web 搜尋整理，給 rag-lab 的 OCR 實驗挑工具。數字都是各家自報或第三方榜單，**沒有繁中分項**。
> 機器：Mac mini M4 24 GB、macOS 26.6.2、ollama 0.40.2、tesseract 5.5.3、pdfplumber 0.11.10 + pypdfium2 5.14。

## 1. Tesseract chi_tra

**下載**（raw URL 已確認 200）
- tessdata_best（最準、LSTM-only、`--oem 1`）：`https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/chi_tra.traineddata`（13 MB）
- tessdata_fast（整數化、最快、略差）：`.../tessdata_fast/main/chi_tra.traineddata`（2.4 MB）
- tessdata（legacy+LSTM，可用 `--oem 0`）：`.../tessdata/main/chi_tra.traineddata`（59 MB）←本機裝的是這份
- 裝完 `tesseract --list-langs`，再 `--print-parameters -l chi_tra` 確認能載入。

**品質**：官方沒給 chi_tra 數字；社群回報的繁中弱點：(a) LSTM 會在漢字之間插空白（`-c preserve_interword_spaces=1`）；(b) 字元集缺「…」；(c) 數字與標點混排掉冒號、括號；(d) 彩色底準確度明顯掉。常用 `-l chi_tra+eng`。

**表格**：tesseract 不認表格。慣例 `--psm 6` + TSV（`image_to_data`），丟 conf<0，依 `top` 聚類成列、列內依 `left` 排序，再對 x 中心做欄聚類。閾值沒有文獻值。M 系列一頁 300 dpi 幾秒。

## 2. PaddleOCR（PP-OCRv5 / PP-StructureV3 / PaddleOCR-VL）

- macOS 只支援 arm64 CPU：`pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/`，再 `pip install -U "paddleocr[doc-parser]"`。
- **繁中是真的有**：PP-OCRv5 單一模型含簡中、拼音、繁中、英、日；官方繁中 rec 0.7472（v4 0.4097）。唯一有繁中分項的傳統 OCR。
- PP-StructureV3 在 Apple Silicon：GitHub #15277 有 M4 24 GB 裝得起來，要 `--use_chart_recognition False`；另一位卡在 table cell NMS `IndexError`。純 CPU、無 Metal。
- PaddleOCR-VL（0.9B）：官方 Apple Silicon 專頁，只在 M4 驗證；要起 `mlx_vlm.server`（違反 rag-lab「不起服務」）。
- 結論：今晚不碰，留到有一整天。

## 3. ollama 視覺模型（0.40.2 全部能跑）

| 模型 | 大小 | 中文 | 表格相關數據 |
|---|---|---|---|
| **glm-ocr**（0.9B） | 2.7 GB | 中英為主 | OmniDocBench v1.5 總分 94.62；Table TEDS 92.6–93.96；`Table Recognition:` 提示直接回 HTML；M4 Max 英文頁約 12 秒/頁；推論記憶體約 2.5 GB |
| deepseek-ocr（3B） | 6.7 GB | 中英 | Table TEDS 84.97；對提示格式敏感 |
| PaddleOCR-VL（社群包） | 936 MB | 109 語 | TEDS 90.89，但 ollama 頁標 input=Text，沒確認吃圖 |
| qwen3-vl:8b / 4b / 2b | 6.2 / 3.3 / 1.9 GB | 32 語 | 通用 VLM；CC-OCR TEDS 8B=60.32 vs PaddleOCR-VL 80.46 |
| qwen2.5vl:7b | 6.0 GB | 有 | 無表格數字 |
| minicpm-v 8b | 5.5 GB | 中英 | OCRBench 強，無表格數字 |
| gemma3:12b / 4b | 8.2 / 3.4 GB | 140 語 | 無 OCR 基準 |
| llama3.2-vision:11b | 7.8 GB | 圖片任務僅英文 | 不適用 |

提醒：ollama 預設 `num_ctx=4096` 對圖片不夠；`OLLAMA_MAX_LOADED_MODELS=1` 換模型付 20–30 秒。

## 4. Apple Vision（ocrmac）

- `pip install ocrmac`，`OCR(img, language_preference=['zh-Hant','en-US'], recognition_level='accurate')`；`fast` 完全認不出中文。
- 回傳 (text, conf, [x,y,w,h]) 正規化、原點左下；是**行框**不是字框；直排不認。
- macOS Sonoma+ 可 `framework='livetext'`（無信心值）。M3 Max 約 0.2 秒/頁。沒有任何公開繁中評測。零依賴、零 port。

## 5. PDF → 圖片（不用 poppler）

- `page.to_image(resolution=300)`（pdfplumber 0.11 底層 pypdfium2）；`.original` 是 PIL；預設 72 dpi 要記得給。
- pypdfium2：`page.render(scale=300/72).to_pil()`；`scale=3` 是 216 dpi 不是 300。
- 300 dpi 安全，小字表格不低於 200。

## 今晚可行的排序

1. **glm-ocr via ollama** —— 一條命令、表格直接 HTML、本地表格分數最高、中文主訓練。
2. **ocrmac + 座標聚類** —— 零服務、秒級，當基線。
3. **tesseract chi_tra（best）+ psm 6 + TSV** —— 最經典的對照組。
4. deepseek-ocr —— 第二個 VLM 對照。
5. qwen3-vl:8b —— 「一個模型讀表又答題」才試。
6. PaddleOCR —— 繁中數據最扎實，但 Mac 坑多、要起服務。

## 沒找到的

- 任何一家的**繁體中文**表格 TEDS 分項（OmniDocBench 不分簡繁）。
- tesseract chi_tra 印刷繁中的公開 CER。
- glm-ocr / deepseek-ocr 在 ollama+Mac 的中文頁秒/頁實測。
- PaddleOCR-VL 社群 ollama 版是否吃圖。
- Apple Vision 對繁中、對表格的量化評測。

主要來源：tesseract-ocr.github.io/tessdoc/Data-Files.html；ollama.com/library/{glm-ocr,deepseek-ocr,qwen3-vl,qwen2.5vl,minicpm-v,gemma3,granite3.2-vision,llama3.2-vision}；paddleocr.ai（PaddleOCR-VL Apple Silicon、PP-OCRv5）；github PaddleOCR discussions #15277；github straussmaximilian/ocrmac；developer.apple.com recognizing-text-in-images；arxiv 2603.10910（GLM-OCR）、2510.14528（PaddleOCR-VL）；benchmarking.nanonets.com/benchmarks/omnidocbench。
