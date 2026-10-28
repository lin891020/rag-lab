# 文件解析工具鏈調查（2026-10-29，給 rag-lab）

> Claude 子代理用 web 搜尋整理。範圍：born-digital PDF、掃描 PDF/影像、DOCX/XLSX/PPTX、HTML、Markdown → 表格要忠實、繁中要對，目標 Mac mini 24 GB 本地。

## 一、比較表

| 工具 | 輸入 | 表格輸出 | CJK/繁中 | Mac 本地 | 授權 | 備註 |
|---|---|---|---|---|---|---|
| **Docling**（IBM） | PDF、影像、DOCX/XLSX/PPTX、HTML、MD | 無損 JSON → MD/HTML/DocTags | OCR 可指定 `zh-Hant`（RapidOCR、tesseract、ocrmac） | 可；layout 走 MPS，**TableFormer 在 MPS 停用只能 CPU**；M3 Max 中位 0.32 s/頁 | MIT | DOCX 直向合併儲存格漏字 bug；XLSX 合併攤平成空字串。https://docling-project.github.io/docling/ |
| **Unstructured** | 20+ 種 | `metadata.text_as_html` | 靠 tesseract/paddle | 可 | Apache-2.0 | docx **自算 colspan/rowspan**；pptx、xlsx 不處理合併 |
| **MinerU 4.0** | PDF、影像、Office、HTML… | 一份文件模型 → 九種 | 簡中強（表格 TEDS 92.1）；繁中未明寫 | Apple Silicon 約 16 GB | Apache-2.0＋附加條件 | https://github.com/opendatalab/MinerU |
| **Marker 2.0** | PDF、影像、Office（轉 PDF） | MD/HTML/JSON | 稱 90 語 | 可 CPU | 權重 OpenRAIL-M | OmniDocBench v1.5 overall 71.3、TEDS 57.9（最差） |
| **pymupdf4llm** | PDF | MD；`table_output="html"` | 無 OCR | 純 CPU 極快 | **AGPL** | `table_strategy` lines_strict/lines/text |
| **xberg（前 Kreuzberg）** | 稱 107 格式 | MD/HTML/JSON/DocTags | tesseract、Paddle、VLM | Homebrew | MIT | 自報 F1 91.0 vs Docling 91.4、快 10–50 倍 |
| **olmOCR 2** | PDF/影像 | MD | 英文導向 | 官方只支援 NVIDIA | Apache | olmOCR-bench 82.4 |
| **Chandra 2** | 影像/PDF | MD/HTML/JSON | 90+ 語 | 5.3B | OpenRAIL-M | olmOCR-bench 85.8（自報） |
| **dots.ocr** | 影像 | HTML | 中文強 | 高解析頁要 30–50 GB，24 GB 吃緊 | MIT | OmniDocBench 88.4 |
| **PaddleOCR-VL 1.5/1.6** | 影像/PDF | HTML、MD | 111 語；**繁中 edit dist 0.048**（自家集） | 0.9B；無官方 ollama/MLX 包 | Apache | v1.5 overall 94.50、TEDS 92.76 |
| **PP-StructureV3** | PDF/影像 | HTML/MD | PP-OCRv5 含繁中 | CPU（Mac 裝機坑） | Apache | overall 86.73、TEDS 81.68（pipeline 類第一） |
| **DeepSeek-OCR / -2** | 影像/PDF | MD+HTML | 中英 | `ollama run deepseek-ocr`（v1） | MIT | v1 87.0、v2 89.2 |
| **GLM-OCR** | 影像/PDF | HTML、MD、JSON schema | 中英 | **0.9B，`ollama run glm-ocr`** | MIT | overall 94.35、**TEDS 93.89** |
| **Mistral OCR**（API） | PDF/影像 | MD+HTML | 多語 | 否 | $2/千頁 | OmniDocBench **78.8**，偏低 |
| **LlamaParse**（API） | 幾乎所有 | MD/JSON | 多語 | 否 | $1.25/千 credit 起 | 小樣本合併儲存格最乾淨 |
| **Azure DI**（API） | PDF/影像/Office | MD（v4.0） | zh-Hant | 否 | Layout $10/千頁 | ParseBench 59.6 |

純 Python 讀取器（表格忠實度）：
- **python-docx**：要 rowspan/colspan 得自己讀 `w:gridSpan`/`w:vMerge`；Unstructured 的 `_convert_table_to_html()` 已做。**docx2python** 有 `duplicate_merged_cells`。
- **mammoth**：docx→HTML；Markdown 輸出已 deprecated。
- **openpyxl**：`ws.merged_cells.ranges`，值只在左上角。
- **python-pptx**：`cell.is_merge_origin / span_height / span_width`。
- **pandas.read_html**：0.24 起處理 colspan/rowspan，值複製到跨越格。

## 二、評測數字

**OmniDocBench v1.5 端到端**（https://github.com/opendatalab/OmniDocBench/tree/v1_5）：

| 方法 | Overall | 表格 TEDS |
|---|---|---|
| PaddleOCR-VL-1.5 | 94.50 | 92.76 |
| GLM-OCR (0.9B) | 94.35 | **93.89** |
| MinerU2.5 | 90.93 | 88.44 |
| DeepSeek-OCR-2 | 89.17 | 85.60 |
| dots.ocr | 88.41 | 86.78 |
| Qwen3-VL-235B | 89.15 | 86.21 |
| GPT-5.2 | 85.75 | 82.76 |
| PP-StructureV3 | 86.73 | 81.68 |
| Mistral OCR | 78.83 | 70.03 |
| Marker-1.8.2 | 71.30 | 57.88 |

v1.5 取消中/英分組；612 頁簡中、290 頁英文，**沒有繁中**。CVPR 2025 原論文：幾乎所有方法中文低於英文。Docling 不在榜。

**olmOCR-bench**（英文為主）：Chandra 83.1、olmOCR 82.4、PaddleOCR-VL 80.0、Marker 76.1、DeepSeek-OCR 75.7、MinerU 75.2、Mistral 72.0。Qwen3-VL-8B 64.6。

**Docling 速度**：4008 頁，M3 Max 中位 0.32 s/頁；關 OCR 省 60%。

## 三、對本地 24 GB Mac 可行的三條路

**路 A：Docling 當骨架，OCR 換 ocrmac/RapidOCR 繁中**——全部格式進 DoclingDocument 存 JSON，再 export。會錯：DOCX 直向合併漏字、XLSX 合併攤平。優點 MIT、一套 API、HybridChunker 現成。

**路 B：格式分流**——born-digital PDF → pymupdf4llm 或 pdfplumber；DOCX → Unstructured `partition_docx`（唯一帶 span 的純 Python 路徑）；XLSX → openpyxl 自己展開；PPTX → python-pptx；HTML → `pandas.read_html`；掃描 → VLM。會錯：五套程式五種 dialect，統一成 HTML 得自己寫；但對「一實驗一資料夾」反而順。

**路 C：VLM 統一吃頁圖——GLM-OCR**——所有來源 rasterize → HTML 表格；Office 先 LibreOffice 轉 PDF。與 `OLLAMA_MAX_LOADED_MODELS=1` 相容。會錯：VLM 對密表會幻覺（codecut 看到「64.4 重複七列」），要配 TEDS 盲測與漂移探針。

**中介表示**：表格用 **HTML**（唯一能表達 span，也是 TEDS 的計分語言），文字用 Markdown，保留無損 JSON。DocTags 只有 granite-docling 產，不當中介。與 03 結論（HTML 對模型讀表沒幫助）不衝突——那是餵模型那一端，這裡講解析保真。

## 四、沒找到的

- **繁中專屬的表格評測**：OmniDocBench 沒有繁中頁；tesseract chi_tra vs PaddleOCR vs Qwen-VL 在繁中 head-to-head 沒有。要自己做。
- Apple Silicon 吞吐量：只有 Docling M3 Max 數字。
- Docling 在 OmniDocBench v1.5 的分數。
- Docling HTML/PPTX backend 的 colspan 處理、mammoth 合併儲存格。
- 2026 大部分比較文都是自報分數彙整。
