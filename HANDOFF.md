# HANDOFF

最後更新：2026-10-28（mini）。

## 現在在哪一步

四個實驗做完：01 切分與檢索、02 JSON 固定輸出、03 表格格式、04 範圍列與寬表（負結果），README 五節都寫滿。
GitHub：`lin891020/rag-lab`（公開，2026-10-28 建）。spec 已核准；04 的計畫在 `~/.claude/plans/jazzy-riding-harbor.md`（本機）。

## 已驗證

- `uv run pytest tests experiments/01-chunking experiments/02-json-output experiments/03-table-format experiments/04-table-lookup`：143 passed（2026-10-28 mini）
- `uv run pre-commit run --all-files` 綠
- 04 `run.py` 820 次呼叫 49 分鐘；漂移 19/20；數字：md+range − md、md+rows − md 兩個模型都分不出，合用在 gpt-oss −.09 分得出；row recall .95。04 README 第 3 節。
- 03 `run.py` 820 次呼叫約 51 分鐘；漂移探針 18/20（差的 2 次都是 gpt-oss t01）；`score.py`、`report.py` 產出 `results/`
- 03 數字：value_ok qwen3 flat .67 → markdown .79、kv .84；gpt-oss flat .73 → markdown .95、kv .87；markdown − flat 與 kv − flat 兩個模型都分得出；html − flat 分不出。全部在 03 README 第 3 節。
- 03 語料：17 張表的 markdown 我逐張對過 PDF 原圖；表九二 文字列曾被誤拆，已修並加測試。
- 02、01 的數字見各自 README；01、02 題庫 Mike 都抽查過。

## 只是寫了、還沒驗證

- 03 題庫 82 題（04 逐字沿用）：正解對 PDF 原圖標；**Mike 還沒抽查**，名單 n03、n17、n39、n44、n53。04 的 `row` 標記也還沒抽查：t01、n54、n17、n28、n26。
- 01 README 第 5 節「正解模糊」兩題（a01→214、a03→表二五～六）只看了 dense 的 miss，沒逐題重審。

## 下一步

1. Mike 抽查 03 的 5 題與 04 的 5 個 `row`；要改就改 `questions.jsonl` 後重跑（可續跑）→ `score.py` → `report.py`。
2. 下一個二選一：05 OCR（spec 原定），或 04 的續集「表註移出表格區塊 + 選欄」（04 README 第 5 節前兩點，最便宜）。開新 session、plan mode。
