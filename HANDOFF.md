# HANDOFF

最後更新：2026-10-28（mini）。

## 現在在哪一步

實驗 01（切分與檢索）做完，README 五節寫滿：`experiments/01-chunking/README.md`。
spec 已核准（2026-10-28，「先直接做，再看怎麼調」），審稿五題的決定補在 spec 最後一節。
計畫在 `~/.claude/plans/jazzy-riding-harbor.md`（本機）。

**還沒推上 GitHub**：Mike 2026-10-28 選「先不建 repo」。本機 main 領先，Air 看不到這個專案。
要建：`gh repo create lin891020/rag-lab --private --source . --push`（spec 說公開，先問）。

## 已驗證

- `uv run pytest tests experiments/01-chunking`：30 passed（2026-10-28 mini）
- `uv run pre-commit run --all-files` 綠；`ruff check` 綠
- `run.py` 全跑約 4 分鐘；重跑第二次 `results/raw.jsonl` 逐字相同；漂移探針 cosine 1.0000
- 模型已拉：bge-m3（digest 7907…6bab）、qwen3:8b（01 沒用到）
- 數字：條文題 dense hit@1000 — fixed1000 .57、article .80、fixed269 .80；article − fixed1000 = +.23 [+.07, +.40]，article − fixed269 = +.00 [−.17, +.13]。表格題 table-whole 最好。全部在 README 第 3 節。

## 只是寫了、還沒驗證

- 題庫 45 題是 Claude 出的。2026-10-28 用 gpt-oss:20b 與 qwen3:8b 模擬抽查（`judge.py`）：正解都答得了、問法自然；**Mike 本人還沒抽查**，抽查名單在 2026-10-28 session 的回報裡（a01、a18、a27、t05、t13）。
- README 第 5 節列的「正解模糊」兩題（a01→214、a03→表二五～六）只看了 dense 的 miss，沒逐題重審。

## 下一步

1. Mike 有空時親自抽查 5 題；不像人問的就改，改了重跑 `run.py`（有快取，幾十秒）再 `report.py`。
2. 決定要不要建 GitHub repo、公開或私有。
3. 02 JSON 固定輸出：開新 session、plan mode、先查前人做法（qwen3:8b 已拉好）。
