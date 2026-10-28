# 異質文件切分策略調查（2026-10-29）

> Claude 子代理用 web 搜尋整理。「二手」= 只從轉述看到的數字。

## 一、策略總表

| 策略 | 做什麼 | 證據／數字 | URL |
|---|---|---|---|
| 固定大小＋重疊 | 按 token 數硬切 | Chroma（5 語料、472 題）：800/400 recall 87.9、IoU 1.4 最差；400/0 recall 89.2。**重疊降 precision/IoU**。Rethinking Chunk Size（2505.21700）：事實型 64–128 tokens 最佳，廣脈絡 512–1024 | https://www.trychroma.com/research/evaluating-chunking ・ https://arxiv.org/abs/2505.21700 |
| 遞迴字元切分 | 段→句→字層層切 | Chroma：**Recursive 200 tokens／無重疊** recall 88.1、IoU 6.9，推薦當預設；策略間 recall 最多差約 9% | 同上 |
| 語意切分 | 相鄰句 embedding 距離過閾值就切 | Vectara（2410.13070）：只有人工拼接語料略勝，**真實語料固定大小一致較好**，「不值得算力」。Chroma：Kamradt 語意 recall 83.6 最差 | https://arxiv.org/abs/2410.13070 |
| Late chunking（Jina） | 整份先過 transformer 再按 chunk pooling | nDCG@10：NFCorpus 23.5→30.0、SciFact 64.2→66.1、FiQA 33.3→33.8；文件越長增益越大 | https://arxiv.org/abs/2409.04701 |
| Contextual Retrieval（Anthropic） | LLM 替每 chunk 寫 50–100 token 脈絡句 | top-20 失敗率 5.7%→3.7%（−35%）→＋BM25 2.9%（−49%）→＋rerank 1.9%（−67%） | https://www.anthropic.com/news/contextual-retrieval |
| 命題切分（Dense X） | LLM 改寫成原子事實句 | 5 個 QA 集勝段落；但 2026 的 36 策略比較把它排在段落切分之下；成本高 | https://arxiv.org/abs/2312.06648 |
| 父子／小到大 | 小塊比對，命中回傳大塊 | 沒找到量化報告。LlamaIndex chunk-size 文：128–2048 中 **1024** faithfulness 最高 | https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5 |
| 版面／元素切分（Unstructured by_title、Docling Hybrid） | 版面元素按節切 | 2026 系統性比較（2603.06976，36 策略×6 領域×5 embedding，含 **bge-m3**）：Paragraph Group nDCG@5 0.459 最高，固定字元 0.244 最差；bge-m3 平均 0.456 最好。結構感知家族整體最強 | https://arxiv.org/html/2603.06976 ・ https://docling-project.github.io/docling/concepts/chunking/ |
| 頁級切分（NVIDIA 2025-06） | 一頁一 chunk | 5 資料集 RAGAS 正確度：**頁級 0.648、SD 0.107 最高最穩**；token 切法 0.603–0.645。實驗裡表格與圖本來就整個保留 | https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/ |
| 混合粒度（MoG） | router 依問題挑大小 | MIRAGE：弱模型最明顯（Qwen1.5 0.456→0.517） | https://arxiv.org/abs/2406.00456 |
| RAPTOR | 聚類→摘要→建樹 | QuALITY＋GPT-4 82.6（基線 55.4） | https://arxiv.org/abs/2401.18059 |
| Agentic／LLM 切分 | LLM 決定邊界 | Chroma LLM chunker recall 91.9 最高但「幾十分鐘」；2026 研究：只有邊際增益 | https://arxiv.org/html/2609.24220v1 |
| 標題鏈前綴 | 每 chunk 前接「文件 > h1 > h2」 | 2608.00824：MRR@5 0.374→0.463（+24%）；distractor 類反降 16.7%；核心發現「拿掉前綴後 LLM 評審一致性 κ 0.45→0.04」 | https://arxiv.org/html/2608.00824 |

## 二、表格怎麼切

**工具實作**
- **Docling HybridChunker**：標題與 caption 進 metadata；`contextualize()` 把標題鏈接回文字再 embed。超長表逐列切，`repeat_table_header=True` 預設每塊重複表頭；`omit_header_on_overflow` 給寬表。預設序列化是 (row, col = value) 三元組。**跨頁表不自動合併**。
- **Unstructured by_title**：Table 永遠獨立 chunk；超過 `max_characters` 拆 TableChunk（**不重複表頭**）；caption 不自動綁。
- **LlamaIndex**：每表一段摘要 IndexNode，RecursiveRetriever 命中摘要下鑽。
- **LangChain MultiVectorRetriever**：摘要進向量庫、原表進 docstore。他們 30 題基準：50-token 切分只 30% 正確；**跨頁表仍是失敗模式**；**表格 chunk 常被文字 chunk 擠掉**，ensemble 把表格權重 0.75 才救回。
- **TableRAG（NeurIPS 2024）**：大表不切，schema 檢索＋cell 檢索。**TableRAG 2025（2506.10380）**：攤平再切會破壞結構，改 SQL＋文字交替；異質文件基準 HeteQA。

**學術證據**
- **Evidence Units（2604.00500）**：以表／圖為種子，把 caption、單位、節標題、相鄰段落（相似 ≥0.40）綁成一個檢索單位。OmniDocBench 1551 題：Recall@1 0.15→0.51、Recall@3 0.69→0.85。只評檢索。**最直接支持「表＋caption＋註綁一起」**。
- **Vision-Guided Chunking（2506.16035）**：多模態按頁批次切、跨批次帶脈絡保跨頁表；0.89 vs 0.78（二手）。

**實務共識**：整張塞得下就不切；塞不下按列群（10–30 列）切，每塊重複表頭＋caption；絕不切在列中；多層表頭展平；精確數值表可改放資料庫。

## 三、給本地小模型 RAG 的建議順序（便宜的先）

1. **遞迴切分 ~200–400 tokens、無重疊，加標題鏈前綴**——重疊只拖累 precision，前綴是零成本 contextual retrieval；bge-m3 是 2603.06976 裡最好的 embedding。
2. **元素切分＋表格整張保留、超長才逐列重表頭**——把 caption 與表註綁進同一 chunk（Evidence Units），直接對 03/04 的失敗模式。
3. **頁級切分當對照組**——一行程式，NVIDIA 說最穩。
4. **表格摘要＋多向量**——qwen3:8b 替每表寫摘要；注意表格 chunk 會被文字擠掉。
5. **Contextual retrieval（LLM 寫脈絡句）**——效果最確定但要預算時間；先用前綴版。
6. **先跳過**：語意切分、late chunking、命題／agentic、RAPTOR／MoG。

## 四、沒找到的

- 沒有基準**單獨量「表註／腳註」怎麼綁**。
- 父子／sentence window 沒有 LlamaIndex／LangChain 的量化比較。
- row-as-chunk＋表頭前綴沒有基準。
- 跨頁表合併：Docling 不做；MinerU、Tensorlake 有，但沒準確率數字。
