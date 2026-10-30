# HANDOFF

最後更新：2026-10-31 00:30（mini）。

## 現在在哪一步

**16-rule-lookup-compute（規則查表接計算，L0／L1／L2）在跑**：`scratchpad/chain16.sh` 2026-10-30 23:51 開跑 `run_read.py`（6 條件 × 2 模型 × 160 = 1,920 次，估 02:30 左右完），接著 score → attribute → report，結束寫 CHAIN16 DONE 到 `scratchpad/chain16.log`。掛了手動照這順序跑（run_read 可續跑，錯誤列會重做）。跑完：README 3–5 節、根 README 一列、playbook 新增「查表接計算」一節。計畫 `docs/superpowers/plans/2026-10-30-16-rule-lookup-compute.md`。
**15-more-questions 做完**（README 五節、根 README、playbook §0、01／07／10 README 各加「15 加倍後」）。
跟 Mike 談定（2026-10-30）：**今晚這批跑完就停，不開新實驗**，11/2 之後再決定方向。
**跑的時候跑測試一律 `uv run pytest tests experiments/*/tests --deselect tests/test_ollama.py --deselect experiments/12-rerank/tests/test_rerankers.py`**（那兩檔會打 bge-m3／載重排器，把跑著的聊天模型踢掉）。
10–14 與更早的交接細節在 `HANDOFF-2026-10-30.md`。

## 已驗證

- 2026-10-30 23:49 15 跑完：閱讀軌 2,560 次 0 錯誤、漂移 28/32（4 次全在 gpt-oss）；新題難度同舊題（full .73／.80 對 .72／.82）。
- 2026-10-31 00:10 15 跑完後才發現：公報新題 80/160 是舊題換句話問（同文件、同答案、引句重疊過半），跟舊題對錯一致 92%（隨機 58%）；法規新題 56/80 問同一條／同一張表。`reasks.py` 排除後重算：25 對裡站得住的分得出只有 bm25 article − fixed1000 +.17、表格攤平 − 逐列 dense −.32；rows 收斂 ±.03；full − −cnpre 合併時分得出、排除後消失。
- 2026-10-31 00:10 07 頁級新規則：舊 40 題 page .53（原 .53），article − page +.30（原 +.28）；審查估的「約 −.10」沒成立。
- 2026-10-31 00:20 全套測試（上面那條 deselect）exit 0。修了兩個全套跑才會壞的測試（09 test_run_retrieval、15 test_same_as_10：函式裡才 import，會拿到 16 的同名模組，改到檔頭）；15 加 `test_reasks.py`；ollama 回退測試的 pyright 型別錯修掉。
- 2026-10-30 16:19 13 做完、14 做完（見各自 README）；12 做完；11、10 做完。

## 只是寫了、還沒驗證

- 16 的閱讀軌還在跑，結果、歸因、報表都還沒看。
- 15 的「排除重複」是跑完才加的檢查，不在事前計畫裡；門檻（引句重疊過半）是我定的，沒有掃過其他門檻。
- 01 條文的 cluster bootstrap 只有 6 群（節），區間不可信，沒拿來下判定。
- 09 題庫剩 145 題沒抽查；05 的 25 題、03／04 標記、07 的 10 題也還沒抽查。

## 下一步

1. 等 CHAIN16 DONE → 讀 `experiments/16-rule-lookup-compute/results/report.md` 與 attribution → README 3–5、根 README、playbook「查表接計算」、HANDOFF、commit、push。
2. 11/2 之後跟 Mike 決定要不要繼續（候補見 playbook「還沒量過的」）。
3. hit@1000 跟 1,000 字塊的耦合要解（切 900 或預算 1,200）。
