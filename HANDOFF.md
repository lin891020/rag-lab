# HANDOFF

最後更新：2026-10-28（mini）。

## 現在在哪一步

實驗 02（JSON 固定輸出）做完，README 五節寫滿：`experiments/02-json-output/README.md`。
01 也完成（Mike 2026-10-28 抽查五題 OK）。spec 已核准。02 的計畫在 `~/.claude/plans/jazzy-riding-harbor.md`（本機）。

**還沒推上 GitHub**：Mike 2026-10-28 選「先不建 repo」，之後同意公開但說先繼續做實驗。
清完
`gh repo create lin891020/rag-lab --public --source . --push`。

## 已驗證

- `uv run pytest tests experiments/01-chunking experiments/02-json-output`：57 passed（2026-10-28 mini，兩種順序都綠）
- `uv run pre-commit run --all-files` 綠
- 02 `run.py` 528 次呼叫約 31 分鐘；漂移探針 16/16 逐字相同；`score.py`、`report.py` 產出 `results/`
- 02 數字：三個開卷條件 parse/schema 全 1.0，value_ok qwen3 .88 [.80, .95]、gpt-oss .95 [.89, 1.0]，配對差距全 +.00；閉卷 .05–.08。全部在 README 第 3 節。
- 01 數字：article − fixed1000 = +.23 [+.07, +.40]，article − fixed269 = +.00；見 01 README。
- 探針（記在 02 README 第 2 節）：gpt-oss:20b 在 ollama 0.40.2 無視 `think:false`，開 schema 會回空字串，要 `think:"low"`。

## 只是寫了、還沒驗證

- 02 題庫 66 題是 Claude 從 01 的題拆出來的，**Mike 還沒抽查**：名單 a01b、a18c、a26b、t09b、t13b（2026-10-28 session 已給他）。
- 01 README 第 5 節「正解模糊」兩題（a01→214、a03→表二五～六）只看了 dense 的 miss，沒逐題重審。

## 下一步

1. Mike 抽查 02 的 5 題；要改就改 `questions.jsonl` 後重跑 `run.py`（可續跑）→ `score.py` → `report.py`。
2. 建公開 repo、push（見上）。
3. 03 OCR，或先做 02 的續集：表格餵 markdown 再量 value_ok（02 README 第 5 節第二點）。開新 session、plan mode。
