# 調查筆記

2026-10-28 深夜 Mike 交代「自己去做更多實驗：不一樣／混合的資料來源、怎麼切、跨頁沒表頭、不同格式怎麼傳遞，先查別人怎麼做」。
四份筆記是 Claude 的子代理用 web 搜尋整理的，數字都註明來源；**沒有繁體中文的公開評測**是四份共同的結論，所以 rag-lab 自己量。

| 筆記 | 一句話 | 餵進哪個實驗 |
|---|---|---|
| [跨頁表格](2026-10-29-cross-page-tables.md) | 開源解析器大多**不合併**（Docling、pymupdf、pdfplumber、Camelot），會合併的（MinerU、Azure、AWS、Doctra）都用同一組幾何規則：前表到頁底、後表在頁頂、中間沒字、欄界對齊、續頁沒標題 | 05 |
| [本地 OCR](2026-10-29-local-ocr-options.md) | 今晚可行的排序：glm-ocr（ollama、表格 TEDS 93.9）> Apple Vision（零安裝、0.8 秒/頁）> tesseract chi_tra；PaddleOCR 在 Mac 要起服務，先不碰 | 06 |
| [解析工具鏈](2026-10-29-parsing-toolchains.md) | 三條路：Docling 當骨架／格式分流各用最會的／VLM 統一吃頁圖；中介表示表格用 HTML（能表達 span）、文字用 Markdown、留無損 JSON | 06、之後的「混合來源」 |
| [切分策略](2026-10-29-chunking-survey.md) | 遞迴 200–400 tokens 無重疊＋標題鏈前綴最便宜；版面元素切分整體最強；表＋caption＋註要綁一起（Evidence Units：Recall@1 .15→.51）；語意切分、late chunking、agentic 先跳過 | 之後的「切分」實驗（01 的續集） |
| [天生就亂的繁中來源](2026-10-29-messy-zhtw-sources.md) | 有標註的繁中亂文件評測集不存在（OmniDocBench 只有 13 頁繁中、JVZY 是合成）；最合用的是**行政院公報法規修正案**（對照表三欄跨頁、開放資料 XML 給 10,438 個 PDF 連結、半年 1,210 筆修正公告）；立法院公報、考選部試題當備用 | 09 |

## 跨四份筆記的共同點

1. **表格要整張保留**，塞不下才逐列切且每塊重複表頭（Docling 預設、3GPP RAG、IntelliRAG）。04 的「只餵 3 列」沒掉準度，跟這一致。
2. **表註是誘餌也是依據**：Evidence Units 把 caption／註綁進檢索單位是對檢索有利；04 看到註裡的數字把模型引走是對生成不利。兩端要分開設計：檢索時綁、生成時標明「附註，非查表值」。
3. **跨頁表格沒有公開的下游實驗**量過「續頁沒表頭」的損失，05 補這個洞。
4. **繁中 OCR 沒有公開評測**（OmniDocBench 不分簡繁、tesseract 無 CER），06 補這個洞。
5. 評測數字幾乎都是自報或榜單彙整；對 POC 的意思是：**挑工具前先用自己的 20 張頁跑一遍 TEDS／格子召回**，不要信榜。
- [2026-10-29 PDF 以外的繁中公開文件來源](2026-10-29-non-pdf-sources.md) — 給 14 用：行政院公報 open data 只有令文是 HTML；各部會主管法規共用系統有要點 HTML 全文、DOC／ODT 附件與歷史版本；臺北市法規系統附件是 ODT；LibreOffice 沒裝
- [2026-10-30 重電業會有哪些文件、什麼格式](2026-10-30-heavy-electrical-doc-types.md) — 從採購制度與施工規範反推的文件家族、公開替身語料清單、實驗 16 的建議
