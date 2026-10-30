# 15 題數要多少才分得出：01／07／10 的「分不出」配對，題庫加倍後重算

## 1. 想回答的問題

01、07、10 有十幾對比較的 95% 區間含 0（「分不出」）。那是真的沒差，還是題太少？題庫加倍（公報 160 → 320、法規條文 30 → 90、表格 15 → 45）後哪些變分得出、哪些收斂成「真的沒差」，以及照每一對實際的不一致率算，要多少題才分得出。

## 2. 怎麼做

**別人怎麼解的**：
- **題數要多少**：Sakai 的 topic set size design（paired t-test／ANOVA／CI 寬度三種算法；像樣的 power 要 200 題以上，IR 慣例的 50–100 題不夠）、Webber／Moffat／Zobel（分得出兩個系統約需 150 topics）。配對的 0／1 結果用 **Connor 1987** 的配對比例公式（Stata `power pairedproportions`、R `pwr_mcnemar` 同一條）：n = ((z_α√p_disc + z_β√(p_disc − d²)) / d)²，p_disc 是兩邊結果不一致的比例，反過來解 d 就是「在 n 題下最小分得出的差距」（MDE）。每一對的 p_disc 都有實際值，所以不用猜。
- **LLM 出題可不可信**：FRAMES（arXiv 2409.12941）LLM 生成題幻覺 >30%、最後用人工；arXiv 2508.11758「Can we Evaluate RAGs with Synthetic Data?」——合成題庫**排檢索設定的名次可靠、排生成器不可靠**。這裡要重算的配對幾乎全是切塊／檢索設定，只有 oracle − full 是生成端。ARES 用 round-trip consistency 濾題；這裡用程式驗引句與數字，加獨立抽查。
- **本機**：09 `tests/test_questions.py`、11 `run_read._number_forms`、10 的 `run_read`／`score`／`report`（複本，加 `--questions`／`--results`）、01／07 的 `run.py`（加 `--questions`／`--out`）、`metrics.paired_diff_ci`。

**先算、先宣告**（`power.py` → `results/power.csv`，`results/power.md`；α .05 雙尾、power .8）。每一對：觀察差距、不一致率、MDE@舊題數、MDE@新題數、新題數下的區間半寬。預測規則：|觀察差距| ≥ MDE@新 → 「會」；半寬 ≤ .03 → 「收斂」（真的沒差，而且界得住）；其餘「不會」。01 的兩列用 30 題估的不一致率、而且標籤規則剛改（見下），標「暫定」，chain_fix 跑完重算。

**事前宣告表**（2026-10-30 09:00 跑 `power.py` 的輸出，跑完不改；01 列暫定）：

| 實驗 | 模型 | 對 | 舊題數 | 觀察差距 | 不一致率 | MDE@舊 | 新題數 | MDE@新 | 新區間半寬 | 偵測觀察差距要幾題 | 預測 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | gpt-oss:20b | +cnpre − fixed | 160 | +0.056 | 0.21 | 0.100 | 320 | 0.071 | ±0.049 | 509 | 不會 |
| 10 | gpt-oss:20b | +rows − fixed | 160 | -0.019 | 0.07 | 0.058 | 320 | 0.041 | ±0.029 | 1533 | 收斂 |
| 10 | gpt-oss:20b | full − -cnpre | 160 | +0.050 | 0.17 | 0.092 | 320 | 0.065 | ±0.046 | 547 | 不會 |
| 10 | gpt-oss:20b | full − -rows | 160 | -0.025 | 0.04 | 0.043 | 320 | 0.030 | ±0.021 | 469 | 收斂 |
| 10 | gpt-oss:20b | oracle − full | 160 | +0.113 | 0.16 | 0.089 | 320 | 0.063 | ±0.042 | 98 | 會 |
| 10 | qwen3:8b | +cnpre − fixed | 160 | +0.044 | 0.22 | 0.103 | 320 | 0.073 | ±0.051 | 895 | 不會 |
| 10 | qwen3:8b | +rows − fixed | 160 | +0.000 | 0.04 | 0.043 | 320 | 0.030 | ±0.021 |  | 收斂 |
| 10 | qwen3:8b | full − -cnpre | 160 | +0.050 | 0.15 | 0.085 | 320 | 0.060 | ±0.042 | 469 | 不會 |
| 10 | qwen3:8b | full − -rows | 160 | -0.013 | 0.01 | 0.025 | 320 | 0.017 | ±0.012 | 626 | 收斂 |
| 10 | qwen3:8b | oracle − full | 160 | +0.056 | 0.12 | 0.076 | 320 | 0.054 | ±0.037 | 292 | 會 |
| 01 | bge-m3 | dense/article − dense/fixed269 | 30 | +0.000 | 0.20 | 0.220 | 90 | 0.130 | ±0.092 |  | 不會 |
| 01 | bge-m3 | bm25/article − bm25/fixed1000 | 30 | +0.133 | 0.33 | 0.284 | 90 | 0.168 | ±0.116 | 145 | 不會 |
| 01 | bge-m3 | bm25/article − bm25/fixed269 | 30 | -0.067 | 0.13 | 0.180 | 90 | 0.106 | ±0.074 | 233 | 不會 |
| 01 | bge-m3 | dense/article − bm25/article | 30 | +0.033 | 0.23 | 0.237 | 90 | 0.141 | ±0.100 | 1646 | 不會 |
| 01 | bge-m3 | dense/table-flat-fixed − dense/table-row | 15 | -0.200 | 0.47 | 0.456 | 30 | 0.336 | ±0.234 | 89 | 不會 |
| 01 | bge-m3 | dense/table-row − dense/table-whole | 15 | -0.067 | 0.20 | 0.299 | 30 | 0.220 | ±0.158 | 351 | 不會 |
| 01 | bge-m3 | bm25/table-flat-fixed − bm25/table-row | 15 | -0.200 | 0.33 | 0.386 | 30 | 0.284 | ±0.194 | 63 | 不會 |
| 01 | bge-m3 | bm25/table-flat-fixed − bm25/table-whole | 15 | -0.200 | 0.20 | 0.299 | 30 | 0.220 | ±0.143 | 37 | 不會 |
| 01 | bge-m3 | bm25/table-row − bm25/table-whole | 15 | +0.000 | 0.13 | 0.244 | 30 | 0.180 | ±0.131 |  | 不會 |
| 07 | bge-m3 | dense/article+context − dense/article | 40 | +0.025 | 0.03 | 0.068 | 90 | 0.046 | ±0.032 | 312 | 不會 |
| 07 | bge-m3 | dense/article − dense/article-noprefix | 40 | +0.000 | 0.00 | 0.000 | 90 | 0.000 | ±0.000 |  | 收斂 |
| 07 | bge-m3 | dense/article − dense/small-to-big | 40 | +0.000 | 0.05 | 0.096 | 90 | 0.065 | ±0.046 |  | 不會 |
| 07 | bge-m3 | hybrid/article − dense/article | 40 | -0.050 | 0.10 | 0.136 | 90 | 0.092 | ±0.065 | 312 | 不會 |
| 07 | bge-m3 | dense/article − bm25/article | 40 | +0.025 | 0.17 | 0.180 | 90 | 0.122 | ±0.086 | 2195 | 不會 |
| 07 | bge-m3 | hybrid/small-to-big − bm25/small-to-big | 40 | +0.025 | 0.07 | 0.118 | 90 | 0.080 | ±0.056 | 940 | 不會 |


**跑前先做的兩個審查決議**（2026-10-29 審查說「要重算」但沒實作；2026-10-30 做掉、各自 commit）：01 固定切的標籤從「塊沾到條文一個字」改成「塊含該條 ≥ min(100 字, 一半)」（`chunkers.span_labels`）；07 頁級標籤改「頁含該條本文的前或後 min(100, 一半)」；兩者的 `metrics.hit_at_budget` 改「第一塊一定讀」（跟 10 的 `budget_evidence` 一致）。01／07 的舊數字留在各自 README，這裡的「舊題」欄用新規則重跑的數字。

**題庫**（寫手是沒看過舊題庫、沒看過任何實驗結果的 agent；我只跑檢查與退件）：
- `questions-gazette.jsonl`：160 題，q161–q320，09 的 30 份公報每份 5–6 題，三個 agent 各 10 份。規則照 09 §2，加一條**數字題的正解數字必須在 quote 裡**（任何寫法：5,000／五千／五○○○／三萬零六百／百分之三十）。不要求 quote 在文件裡唯一（舊題三分之一本來就重複；`is_gold` 任一塊含 quote 就算），只計數。配額：含「萬」≥ 10、阿拉伯數字 ≥ 15。
- `questions-law.jsonl`：01 語料第二章（第 16–110 條）條文題 50（a41–a90）、該章引用的 24 張表表格題 30（t18–t47），一個 agent。
- 檢查：`tests/test_questions_gazette.py`、`tests/test_questions_law.py`（退件率記下面）；抽查：沒看過題目的 agent 從頁面截圖獨立作答，公報 20 題（seed 15）、法規 10 題。
- 寫手簡報裡的三件提醒：草案的現行欄是誘餌、問修正後；一筆夾多部法規的只出第一部；quote 不要從條號那一行抄起（article-cn 的塊文沒有條號行，HANDOFF 的 q061／q067）。

**跑**（`chain15.sh`，等 chain_fix 結束才碰 ollama）：
- 公報閱讀軌 `run_read.py`：closed-book、oracle、fixed、+cnpre、+rows、full、−cnpre、−rows（8）× qwen3:8b、gpt-oss:20b × 160 = 2,560 次；依據組法、提示、schema、預算逐位元同 10（`tests/test_same_as_10.py`）。舊 160 題（含 chain_fix 重跑的 oracle）用 15 的 `score.py` 重算（`results/rescored-old/`），單位別名只准增不准改。
- 法規檢索軌：01、07 的 `run.py --questions`（01 的兩個題庫 + 新題庫）`--out results/law01/`、`law07/`，無模型。
- 合併 `merge.py`：每一對四欄——舊題、新題、合併（按題 bootstrap，2000 次、seed 0、兩臂同抽）、合併（**cluster bootstrap**：放回抽 30 份文件；法規以 gold 條所在的節為群、表各自一群）；對照事前預測標「如預期／意外」。約 25 對 × 2 模型 × 1 尺，不校正——配對事前宣告、合併是事前計畫，不是逐步檢定。

**五條規矩**：雜訊底線 closed-book（公報）／random（法規）；盲測基線 fixed（公報）、fixed269（01）、article（07）；bootstrap 95% 配對差距 + cluster 版；版本表（新舊兩組）；漂移探針每格 2 題，新舊分開報。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-30（出題、程式）、閱讀軌跑完填／mini.local |
| ollama／bge-m3 | 0.40.2／`790764642607` |
| qwen3:8b／gpt-oss:20b digest（舊 160 題，10） | `500a1f067a9f`／`f38aa0c53da5` |
| qwen3:8b／gpt-oss:20b digest（新 160 題） | （跑完填） |
| Python／pypdf／pdfplumber／PyMuPDF | 3.12.15／6.21.3／0.11.10／2.0 |

## 3. 數字

（跑完填：事前宣告表、題庫統計與退件率、抽查、合併表）

## 4. 結論

（跑完填）

## 5. 沒做到的、下次想試的

（跑完填）
