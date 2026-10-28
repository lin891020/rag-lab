# 07 切分續集：標題鏈、脈絡句、頁級、父子、混合檢索，哪個讓條文找得更準

## 1. 想回答的問題

01 的結論是「小塊贏，不是照條贏」（一條一塊 .80 vs 固定 1000 字 .57，hit@1000）。切分調查（`docs/research/2026-10-29-chunking-survey.md`）列出幾個便宜的招：標題鏈前綴（零成本的 contextual retrieval）、LLM 寫脈絡句（Anthropic 的 contextual retrieval，−35%～−49% 失敗率）、頁級切塊（NVIDIA 說最穩）、父子（小塊找、整條回）、dense + BM25 混合。在同一份法規、同一組題、本地模型上，這些招各加多少？

## 2. 怎麼做

**別人怎麼解的**：Chroma（2024）——遞迴 200 tokens 無重疊最穩，重疊只拖累 precision；Anthropic（2024）——每 chunk 由 LLM 寫 50–100 token 脈絡句再 embed，top-20 失敗率 5.7%→3.7%，加 BM25 2.9%；2608.00824——標題鏈前綴 MRR@5 +24%；NVIDIA（2025）——頁級切塊 RAGAS 0.648 最穩；2603.06976（36 策略 × 5 embedding，含 bge-m3）——結構感知切分整體最強、bge-m3 是最好的 embedding；LlamaIndex／LangChain 的父子只有教學沒數字。**沒找到**本地 8B 模型寫脈絡句的量測。

**資料與題庫**：01 的 `law.pdf`（283 頁、1014 條）與 109 張表；01 的 30 條文題 + 保留的 10 條文題 = **40 題**（01 README 說保留題「留給第二輪」，這是第二輪）。表格題不做：這些招是文字的。

**條件**（`run.py`；表格永遠是 table-whole，01 表格題的贏家）：

| 切法 | 做法 | 對應的招 |
|---|---|---|
| article | 01 的贏家：一條一塊，前置「章 / 節 / 第 N 條」 | 基線 |
| article-noprefix | 同樣的塊，拿掉那一行 | 標題鏈前綴（反向：拿掉看掉多少） |
| article+context | 每塊前面加 qwen3:8b 寫的一兩句（`context.py`，≤ 60 字，給它章節路徑與該條，不給全文） | contextual retrieval，本地版 |
| page | 一頁一塊（pypdf 頁文字），標籤 = 頁上出現的條 + 從上一頁延續的條 | 頁級切塊 |
| small-to-big | 索引 269 字固定塊（01 的 fixed269），命中後回它所在的整條 | 父子 |

**檢索**：dense（bge-m3）、bm25（字二元組）、**hybrid**（兩者 RRF，k=60、各取前 100）、random（雜訊底線）。指標同 01：hit@5、hit@1000、hit@2000，結論看 hit@1000（父子條件的預算算整條的長度）。

**五條規矩**：雜訊底線 = random；盲測基線 = article／dense（01 的數字）；區間 = bootstrap 95%、配對差距；記版本 = 下表；漂移探針 = 重抽 20 塊直接打 ollama 算向量，與快取的 cosine。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-29／mini.local |
| ollama | 0.40.2 |
| bge-m3 digest | `sha256:7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab` |
| qwen3:8b（脈絡句） | `sha256:500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`，think off、temperature 0、num_predict 160 |
| Python／pypdf | 3.12.15／6.21.3 |

## 3. 數字

RESULTS_PLACEHOLDER

## 4. 結論

CONCLUSION_PLACEHOLDER

## 5. 沒做到的、下次想試的

- **40 題**，區間寬；01 就說過要分出 ±.05 得 150 題以上。
- 脈絡句只給模型章節路徑與該條，不是 Anthropic 那種「全文 + 該段」（255k 字塞不進 num_ctx；要做得分章餵）。
- 沒試「脈絡句只進 BM25 不進向量」或反過來；沒試 rerank。
- page 的標籤規則「上一頁延續的條」假設條文連續；跨頁的表（05）在 page 條件裡是整張表塊，不受影響。
- 父子只試了 269 字一種子塊大小。
