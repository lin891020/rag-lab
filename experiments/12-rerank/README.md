# 12 重排序值不值得它的時間：dense 前 50 塊再排一次，找得到、答得對各多多少

## 1. 想回答的問題

bge-m3 撈回前 50 塊之後，再用一個重排器排一次（cross-encoder、生成式 pointwise、LLM listwise 各一種），「排第一的是不是正解」「1,000 字預算內有沒有正解」「答對率」各變多少、每題多花幾秒？第一階段已經強（09 最好的格 hit@5 .93–.98）時還值不值得？

## 2. 怎麼做

**別人怎麼解的**：
- **cross-encoder 重排是標配**：BAAI 自己的 RAG 評測說 bge-reranker-v2 配 bge-m3 端到端最好；韓文法規檢索（EVE-Rerank-1.0-Legal 的模型卡）用 bge-reranker-v2-m3 重排前 20，基礎版只 +.03——**第一階段已經強時提升是小的**。COLIEE 2026（arXiv 2609.26237）drop-one 說 cross-encoder 貢獻最大，但那是案例檢索、第一階段弱。→ 預期這裡是小提升，主要在 hit@1 與預算內。
- **LLM 重排**：RankGPT（arXiv 2304.09542）listwise 用 GPT-4；Rank-without-GPT（arXiv 2312.02969）開源 7B listwise 跟同大小 pointwise 在 TREC-DL 只差一點；Qwen3-Reranker-0.6B 是生成式 pointwise（yes／no token 的機率），CMTEB-R 71.3。→ 三種各試一個。
- **ollama 沒有 rerank API**（issue #4510 開著），所以 cross-encoder 走 Python：新依賴 `sentence-transformers`＋`transformers`＋`torch`，放 optional group `rerank`（`uv sync --group rerank`），模型下載到 `~/.cache/huggingface`。LLM listwise 照舊走 ollama。
- **本機**：07 試過 dense＋BM25 混合（RRF）沒幫助——那是「合併兩個排名」不是重排，這裡當對照組；10 的 `run_retrieval`（`DenseIndex`、`hit_at_k`／`hit_at_budget`／`first_hit_rank`）、09 的 `budget_evidence`、10 的 `score.judge`（新尺）。

**語料與題庫**：09 的 30 份公報、160 題；文字讀 09／10 的路線，不複製。

**格**（`run_rerank.py`）：
- 切塊兩格：`full`（ocrrows／article-cn+pre／前綴，10 的）、`fixed`（ocrrows／fixed-1000／前綴，09 有 10 份沒有「條」只能靠它）。
- 第一階段：dense bge-m3 前 50（`DEPTH`）。
- 重排器五種（`rerankers.py`）：

| 重排器 | 做什麼 | 跑在哪 |
|---|---|---|
| none | dense 原序 | — |
| bm25 | 07 的 RRF（dense 全排名＋字二元組 BM25 全排名合併）取前 50；對照：合併不是重排 | CPU |
| bge | `BAAI/bge-reranker-v2-m3` cross-encoder，前 50 每塊打分 | torch／MPS，分數快取 `data/12-rerank/` |
| qwen3-rr | `Qwen/Qwen3-Reranker-0.6B`，官方模板、P(yes) 當分數，前 50 | 同上 |
| listwise | qwen3:8b `think:false`，前 20 塊各截 400 字，RankGPT 式「排順序」、JSON 整數陣列；缺號照原序補、重複去掉，不合格的輸出計數；21–50 名保持原序 | ollama |

- 檢索指標：hit@1、MRR、hit@5、hit@1000、hit@2000、**預算內有正解**（照 09 `budget_evidence` 塞：整塊放得進 1,000 字才進，一塊都放不進就截第一塊——這是閱讀軌真的看到的）、每題秒數。bootstrap 95%、每個重排器 − none 配對差距。
- 閱讀軌（`run_read.py`）：只做 `full` 格 × {none, bge, qwen3-rr, listwise} × qwen3:8b／gpt-oss:20b × 160 題 = 1,280 次；提示是 09 的 P0 逐字（`tests/test_prompts.py`）、尺是 10 的。none 應該跟 10 的 full 在漂移範圍內一樣，用來對帳。
- 漂移：每格重排 2 題再比順序；閱讀每格重打 2 題。

**五條規矩**：雜訊底線＝none 已含（dense 原序）＋09 的 random；盲測基線 none；bootstrap 95% + 配對差距；版本表（下）；漂移探針。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-29／mini.local |
| ollama／bge-m3／qwen3:8b | 0.40.2／`790764642607`／`500a1f067a9f` |
| torch／sentence-transformers／transformers | 2.15.0（MPS）／6.2.0／5.21.0 |
| bge-reranker-v2-m3／Qwen3-Reranker-0.6B | HF `BAAI/bge-reranker-v2-m3`、`Qwen/Qwen3-Reranker-0.6B`（2026-10-29 快取版） |
| gpt-oss:20b digest | `f38aa0c53da5` |

**預期**（事先宣告）：hit@5 全部貼頂（.93–.98）分不出；hit@1 none 約 .7，bge／qwen3-rr +.05–.10，listwise 跟 bge 分不出但慢 20 倍以上；預算內有正解 +.05 以內；答對率 ±.03 分不出。

## 3. 數字

檢索軌 2026-10-30 03:21–11:10（兩格 × 五重排器 × 160 題；重排漂移 12/12 同序）；閱讀軌 11:10–13:16，1,280 次、0 錯誤、漂移 15/16。全表在 `results/report.md`。

### 檢索（每題秒數是重排的實測）

| 格 | 重排器 | hit@1 | MRR | 預算內有正解 | 秒/題 |
|---|---|---|---|---|---|
| full | none | .76 [.69, .83] | .86 | .82 [.76, .88] | 0 |
| full | bm25（RRF） | .79 | .87 | .86 | 0 |
| full | **bge** | **.93 [.88, .96]** | **.95** | **.93 [.89, .97]** | 2.9 |
| full | qwen3-rr | .85 | .91 | .88 | 36.8 |
| full | listwise | .78 | .85 | .81 | 31.4 |
| fixed | none | .69 [.62, .76] | .81 | .69 [.62, .76] | 0 |
| fixed | bm25 | .73 | .85 | .73 | 0 |
| fixed | **bge** | **.90 [.85, .94]** | **.94** | **.90 [.85, .94]** | 25.7 |
| fixed | qwen3-rr | .79 | .88 | .79 | 39.2 |
| fixed | listwise | .55 [.47, .62] | .68 | .56 [.48, .63] | 40.4 |

配對差距（− none，✓ 分得出）：

| 重排器 | full hit@1 | full 預算內有正解 | fixed hit@1 | fixed 預算內有正解 |
|---|---|---|---|---|
| bm25 | +.03 [−.01, +.07] | +.03 [−.01, +.07] | +.04 [−.01, +.08] | +.04 [−.01, +.08] |
| bge | **+.16 [+.11, +.23]✓** | **+.11 [+.05, +.16]✓** | **+.21 [+.14, +.28]✓** | **+.21 [+.14, +.28]✓** |
| qwen3-rr | +.09 [+.03, +.16]✓ | +.05 [+.00, +.11] | +.10 [+.03, +.17]✓ | +.10 [+.03, +.17]✓ |
| listwise | +.02 [−.06, +.10] | −.02 [−.08, +.04] | **−.14 [−.23, −.06]✓** | **−.14 [−.23, −.06]✓** |

hit@5 全部 .93–.99 貼頂、分不出（預期中）；hit@1000 只有 fixed 格的 bge +.08 分得出。

### 閱讀（full 格）

| 模型 | 重排器 | 嚴格 | 寬鬆 | 依據含正解 | − none（嚴格） |
|---|---|---|---|---|---|
| qwen3:8b | none | .81 [.75, .88] | .81 | .82 | — |
| qwen3:8b | bge | **.86 [.81, .91]** | .86 | .93 | **+.05 [+.01, +.10]✓** |
| qwen3:8b | qwen3-rr | .84 | .84 | .88 | +.03 [−.02, +.07] |
| qwen3:8b | listwise | .78 | .78 | .81 | −.03 [−.09, +.03] |
| gpt-oss:20b | none | .73 [.66, .80] | .83 | .82 | — |
| gpt-oss:20b | bge | **.79 [.73, .86]** | .91 | .93 | **+.06 [+.01, +.11]✓** |
| gpt-oss:20b | qwen3-rr | .77 | .88 | .88 | +.04 [−.02, +.09] |
| gpt-oss:20b | listwise | .74 | .83 | .81 | +.01 [−.06, +.06] |

none 跟 10 的 full（.82／.72）在漂移範圍內對得上（.81／.73）。

## 4. 結論

**cross-encoder 重排（bge-reranker-v2-m3）值得掛：預算內有正解 +.11／+.21、答對率 +.05／+.06，兩個模型都分得出，一題 3 秒；其他三種不值得——RRF 合併 ±.03、Qwen3 生成式重排 +.05 但一題 37 秒、LLM 列表式重排 ±0 到 −.14 還一題 31–40 秒。** 第一階段越弱（fixed 格）重排救得越多（+.21 vs +.11），跟文獻（EVE-Rerank、COLIEE）一致。答對率的收益比檢索的小一半：撈到了還要讀對，依據含正解 .82 → .93 只換到答對 +.05。

事前宣告對了三件（hit@5 貼頂、bge 在 hit@1 與預算內有正解有感、答對率小）、錯了兩件：bge 的提升不是「+.05–.10」是 +.11–.21；listwise 不是「跟 bge 分不出」是在 fixed 格明顯變差（20 塊各截 400 字，塊長的格它看不到關鍵句）。另外重排器在長塊上慢三到九倍（bge 2.9 → 25.7 秒），估時程要按格算。

## 5. 沒做到的、下次想試的

- 只重排前 50（Qwen3 太慢），前 100 沒試；listwise 窗 20 塊、每塊 400 字是為了塞 context，換成 10 塊 × 800 字沒試。
- 閱讀軌只做 full 格；fixed 格 +.21 的檢索收益到答對率是多少沒量。
- bge 的分數快取在 `data/12-rerank/`，不進 git；重跑要 `uv sync --group rerank`。
- Qwen3-Reranker-0.6B 的提示用官方模板（英文 instruction），中文 instruction 沒試。
- 兩種 LLM 重排都慢到上不了線（31–40 秒／題）；真要用得批次或換 4B 以下。
- 10／12 的 `article_chunks` 丟「第 N 條」行（審查發現，2 題永遠不命中），兩個實驗都沒改，數字不受影響只是那 2 題恆錯。
