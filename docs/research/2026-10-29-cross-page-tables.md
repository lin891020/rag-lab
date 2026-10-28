# 跨頁表格（續頁不重複表頭）：別人怎麼解的（2026-10-29 調查）

> Claude 子代理用 web 搜尋整理。實驗 05 的「別人怎麼解的」就是這份。

## 一、開源／商用解析器

- **MinerU**（規則式，預設開，`MINERU_TABLE_MERGE_ENABLE=0` 關）：只看「前頁最後一個 block 是表、本頁第一個 block 是表」；本頁表若有 caption 必須含「續／續表／(continued)」才合併；前頁表有 footnote 就擋（有續標記放寬到 1 條）；兩表寬度差 ≥10% 不合併；HTML 欄數（含 colspan）相等，或「前表末列 vs 後表第一資料列」欄數相等；`detect_table_headers` 逐列比對最多 5 列文字＋span 找重複表頭刪掉再接。<https://github.com/opendatalab/MinerU/blob/mineru-2.7.6-released/mineru/utils/table_merge.py>
- **MinerU-Popo**（2026-05，模型式）：「表格截斷復原」獨立子任務，候選對＝頁尾最後元素＋次頁第一元素，先啟發式篩選（續標記、bbox 寬度、欄數），再由模型判「是否同一表」＋「邊界上每欄要不要接儲存格」。165 題、1,317 個儲存格合併單元，準確率 92.7%。<https://arxiv.org/html/2605.24973>
- **Docling**：不合併，每頁各一個 `TableItem`，續頁第一列會被標成表頭（issue #2976，仍 open）。#2060 原型：頁 N 最後一表與頁 N+1 第一表，門檻 `BOTTOM_TH=0.20, TOP_TH=0.80`，不比欄數，交 LLM 判斷，再 `_dedup` 刪重複表頭。<https://github.com/docling-project/docling/issues/2976>
- **Marker**（`--use_llm` 才做）：幾何篩選——較大那張表高度 >60% 頁高、兩頁各只一張表、列數差 <5 或欄數差 <2；通過才送 LLM，prompt 規定「表 2 沒表頭且其列延續表 1」才接。使用者抱怨常誤合併（#608）。<https://github.com/datalab-to/marker/blob/master/marker/processors/llm/llm_table_merge.py>
- **pymupdf4llm / PyMuPDF**：不合併；維護者立場「table finder 不可能比你更知道」。PR #470 曾加「外框 x 範圍相同＋垂直相鄰＋文字落得進前表欄格」就接，但 503 頁 benchmark −0.2、6 頁有 5 頁退步，被拿掉。<https://github.com/pymupdf/pymupdf4llm/pull/470>
- **pdfplumber**：無功能。作者建議切點可預測就用 `explicit_horizontal_lines`，否則逐頁抽再自己接。<https://github.com/jsvine/pdfplumber/discussions/1086>
- **Camelot**：無功能，社群 `pages='all'` 後按欄數分組 `pd.concat`。
- **Unstructured**：`partition_pdf` 不做；平台有「Chunk by Table Merging」節點，相鄰頁、有 `text_as_html` 的 Table，由 LLM 決定。<https://docs.unstructured.io/api-reference/workflow/nodes/enhancement/chunker-chunk-by-table-merging>
- **LlamaParse**：`merge_continued_tables=true`（agentic tier），掛 `merged_from_pages`，演算法未公開。
- **Azure Document Intelligence**：Layout 不合併；官方 blog 啟發式——兩表之間只允許 pageHeader/pageFooter/pageNumber 段落、欄數相同即合併（不同直接 `ValueError`）；續頁有表頭就刪。<https://techcommunity.microsoft.com/blog/azure-ai-services-blog/a-heuristic-method-of-merging-cross-page-tables-based-on-document-intelligence-l/4118126>
- **AWS Textract**（`pipeline_merge_tables`）：(1) 頁尾表到頁底、頁首到次頁表之間不能有其他文字；(2) 欄數相同 **或** 第一列文字完全相同；(3) 表寬與左邊界差 < (100−accuracy)%。<https://github.com/aws-samples/amazon-textract-response-parser/blob/master/src-python/trp/t_tables.py>
- **小工具**：Doctra——前表底 ≥80% 頁高、後表頂 ≤15%、水平重疊 ≥50%、寬度差 ≤20%、每欄 x 10 px 內對齊且 ≥60% 欄對齊，加權信心 ≥0.65。<https://ademboukhris457.github.io/Doctra/user-guide/features/split-table-merging.html>；table-stitcher——表頭 token Jaccard ≥0.6、無表頭片段只接前一張。<https://pypi.org/project/table-stitcher/>

## 二、論文

- **PubTables-v2**（2025-12）：第一個有跨頁標註的資料集；24,862 張表有 9,492 張跨頁（約 38%）。任務「兩頁影像並排 → 最後一表是否延續」，ViT-B-16 F1 0.991、ResNet-50 0.979；250 筆訓練就 F1 0.86。合併 dots.ocr 逐頁結果（欄數相同就垂直接），GriTS_Con 0.577→0.684。<https://arxiv.org/html/2512.10888v3>
- **POTATR**（2026-06）：同一分類器（recall 0.995）＋垂直串接，GriTS_Con 0.671→0.827，錯誤減 47%。<https://arxiv.org/html/2606.09788v1>
- **MonkeyOCR v1.5**：頁首文字插在表中間時 MinerU2.5、PaddleOCR-VL 都接不回去。
- **RAGFlow #18569**（繁中 Dell 文件）：跨頁長段落儲存格被拆成 13 列。

## 三、社群啟發式

- 頁底／頁頂門檻 0.8／0.2＋末列欄數＝首列欄數，不處理表頭。
- 「全部欄對齊且第一欄空白 → 是前一列的延續列」（Docling #2976 回報者）。
- 表頭重複偵測：MinerU 逐列比文字＋span；AWS 第一列完全相等；table-stitcher Jaccard。
- forward-fill 只在「合併儲存格變空白」被提到，沒人當跨頁判據。

## 四、不合併會怎樣（下游證據）

- **3GPP 規格 RAG**（2024，278 題、21,824 張表、含 bge-m3）：每個 chunk 重複表頭的 top-5 檢索準確率高於不重複，列級 chunk 優於表級；只有長條圖。<https://arxiv.org/html/2408.17008>
- **土耳其文 RAG**（2026）：Docling 版面感知切塊讓表格題比固定長度高 17.0 分、比語意切塊高 8.7 分，原因是整張表不被切開。<https://arxiv.org/pdf/2608.26192>
- 實務回報：IntelliRAG PR #26 把表頭補進每個續 chunk，token 只多約 3%。

## 最便宜的作法（pdfplumber 幾何）

1. 只看「頁 N 最後一張表、頁 N+1 第一張表」這一對（MinerU、AWS、Docling #2060）。
2. 前表 bbox 底 ≥ 0.8 頁高、後表頂 ≤ 0.2 頁高。
3. 兩表之間除頁碼／頁首頁尾外不得有文字（Azure、AWS）。
4. 欄界比對：各欄 x 邊界 10 px 內對齊的欄 ≥60%、表寬差 <10%（Doctra、MinerU）。
5. 欄數相等 **或** 首列文字與前表表頭相等（AWS）；相等就刪續頁表頭。
6. 續頁第一列第一欄空白 → 併進前表最後一列（Docling #2976；MinerU-Popo）。
7. caption 只准在第一頁；續頁有 caption 必須含「續」（MinerU）。
8. 前表有 footnote 就不合併（MinerU）。
9. 合併後記 `merged_from_pages`。
10. 規則式至少做到「欄數相等就接」；要更高再考慮 ViT 二分類。

## 沒找到的

- GTE、TableSense、ICDAR 2013/2019：沒有跨頁合併的討論或評測。
- 「續頁不重複表頭」被單獨量化的下游實驗：沒有（3GPP 與土耳其研究量的是「表頭重複／整表不切」）。
- MinerU 2.2.0 原始演算法、LlamaParse 與 Tensorlake 內部規則：未公開。
- 中文法規 PDF 的跨頁表格專題：只有 CSDN 通用教學。
