# HANDOFF

最後更新：2026-10-29 16:30（mini）。

## 現在在哪一步

10-fix-ablation 跑到一半：把 09 的三個修法拆開量（加一個／關一個，13 條件 × 2 模型 × 160 題 = 4,160 次），評分尺補了公報單位、加寬鬆分、數字按寫法分組；09 用新尺重算完（oracle .87／.83，寬鬆 .87／.93）。閱讀軌 2026-10-29 15:01 在 mini 背景啟動，約 21:00 跑完，可續跑（`uv run python experiments/10-fix-ablation/run_read.py`）。計畫經 Mike 核准（`~/.claude/plans/jazzy-riding-harbor.md`）。
跑完：`score.py`、`report.py`，填 README 第 3 節閱讀軌、第 4、5 節，根 README 加 10 那列。09 做完（README 五節齊）。
11-abstain（找不到要說找不到）程式寫完、煙霧測試過，`scratchpad/chain11.sh` 會在 10 跑完後自動開跑 3,200 次（約 4.5 小時，到隔天凌晨）；掛了就手動 `uv run python experiments/11-abstain/run_read.py`（可續跑）。
12-rerank（重排序）程式也寫完、煙霧測試過，`scratchpad/chain12.sh` 在 11 跑完後接著跑 `run_rerank.py`（兩格 × 五重排器，需 `uv sync --group rerank`）再 `run_read.py`（1,280 次），估到隔天上午；掛了手動跑，都可續跑。
`docs/2026-10-29-real-data-playbook.md`（01–10 整理成一條流程）初稿已推，10、11、12 的數字跑完補。

## 已驗證

- 12：11 個測試綠；煙霧測試 3 題 × 5 重排器（bge／Qwen3-Reranker 在 MPS 上跑得動、listwise 走 ollama）、漂移 6/6；新依賴 optional group `rerank` 裝好（torch 2.15 MPS）。
- 11：14 個測試綠（P0 逐字同 09、leak 抓中文／阿拉伯數字、濾網去空白比對）；煙霧測試 2 題 × 2 提示 × 5 依據 20 次 OK（P1 在沒答案的依據上 6/6 棄答）。
- 10 `extract.py` ocrrows 30 份；`score.py` 用新尺重算 09（`results/09-rescored/`）；`tests/test_run_read` 證明 10 組的依據跟 09 餵模型的字數相同；煙霧測試 2 題 × 3 條件 OK。
- 09 語料：`build_corpus.py` seed 0 掃 233 筆湊齊 30 份（草案 15／令 15、地方 14、掃描 4、亂碼 1、旋轉附表 7），`corpus/manifest.json`、`corpus/health.csv`。
- 09 `extract.py` 三路線跑完：OCR 42 頁（glm-ocr，快取在 `data/09-messy-real/ocr`）、對照表 rows 在 15 份草案的 7 份找到三欄表。
- 09 `run_retrieval.py`：72 格 × 160 題，random .00–.01，向量 cache cosine 1.0000；base／article .00（160/160 切不出）、ocr／article-cn+pre dense hit@2000 .89 [.83, .93]；修法配對差距：中文條號 +.55、前言留塊 +.23、OCR +.11 分得出，fixed −.03、rows +.01、前綴 +.01 分不出（hit@5 +.04 分得出）。`attribute.py` 錯因在 `results/attribution.csv`。
- 09 `run_read.py`：160 題 × 5 條件 × 2 模型 = 1,620 次 141 分鐘 0 錯誤、漂移 19/20；base 等於閉卷（.05／.04，差距含 0），fixed .48／.42，all .71／.64，oracle .84／.82；fixed − base +.43／+.39、all − fixed +.23／+.21 都分得出。
- `uv run pytest tests experiments/*/tests` 與 `pre-commit run --all-files`：見本次 commit 訊息的數字（2026-10-29 13:40 mini）。
- 09 題庫抽查 15 題（seed 9 名單）：Mike 太累改由一個沒看過題目的 agent 從頁面截圖獨立作答再比對，15/15 答案對、0 錯；它指出 q037 與 q144 的 quote 撐不滿答案（同句在別處先出現／只涵蓋 7 條裡 5 條）、q067 的附表條次要不要算沒寫明，答案本身都對，quote 沒改（改了會動檢索命中）。
- 01–08 見各自 README；HANDOFF-2026-10-28 的「已驗證」照舊。

## 只是寫了、還沒驗證

- 09 題庫剩下 145 題沒抽查（15 題已由獨立 agent 核過，見「已驗證」）。05 的 25 題、03／04 標記、07 的 10 題也都還沒抽查。
- 09 閱讀軌的 all 條件沒含 OCR（rows 路線沒接 ocr 的頁），掃描／亂碼那 20 題在閱讀軌永遠沒正解；「OCR 對閱讀值多少」只有檢索軌的 +.11 可推。
- 09 的單位評分（06 的 judge）把「月／個月」「日／工作日」「人／名」與 gpt-oss 的空單位判錯，寬鬆分 oracle .89／.93；沒改評分，README 兩個數都報。
- 08 對照表的修法這次只修了三條（中文條號、前言留塊、OCR），無框線「x 位置當欄線」沒遇到也沒修。

## 下一步

1. 等 10 的閱讀軌跑完（`results/read/raw.jsonl` 4,160 列）→ score、report、README 3–5 節、根 README、playbook 補數字。
2. 等 11 跑完（3,200 列）→ 同上，playbook 第 5 節加「找不到要說找不到」。
3. 等 12 跑完（rerank 1,600 列、read 1,280 列）→ 同上，playbook 第 4 節加重排。之後候選：要點／附表切法、非 PDF 來源、文件改版。
4. hit@1000 跟 1,000 字塊的耦合要解（切 900 或預算 1,200），之後的實驗都受影響。
