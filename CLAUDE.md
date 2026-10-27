# rag-lab

長期的個人實驗室：練「資料進 → 模型出」這條線上的技術（文件切分與檢索、固定格式輸出、OCR、之後的新方法）。
**產出是實驗紀錄，不是產品。** 每個實驗一個資料夾、一頁 README，程式只是產生數字的工具。
背景、邊界、談定的事全在 `docs/superpowers/specs/2026-10-28-rag-lab-design.md`，開工前先讀。

## 怎麼跑

```bash
uv sync
uv run python experiments/<NN-name>/run.py   # 每個實驗自己帶程式
```

## 這個專案獨有的

- **一次只開一個實驗**，做完（README 五節寫滿）再開下一個。現在在哪個看 `HANDOFF.md`。
- `experiments/NN-name/`：程式、題庫、結果、README 都在裡面，實驗之間不共用程式，重複就重複。
  `common/` 只放真的不會變的（連 ollama、讀 PDF）；要改它得有理由寫在 commit。
- 小資料進 git（法規 PDF、手工題庫、結果 CSV，單檔 ≤10 MB）；大檔放 `data/`，不進 git。
  這是對家裡層「資料集不走 git」的明示例外，理由在 spec。
- 模型一律走 host 的 ollama `127.0.0.1:11434`；向量 `bge-m3`、聊天 `qwen3:8b`，`gpt-oss:20b` 當對照。
- 不開 port；向量庫用檔案型，不起服務。
- **不收 DXF 產圖與排版**，要練另開專案。
- 每個實驗開工前先用 plan mode 提計畫，計畫要有「別人怎麼解的」。評測照 Mike 的五條規矩（雜訊底線、bootstrap 區間、盲測基線、記版本、漂移探針）。
- 文件中文、程式英文。回報白話、一次一個重點（Mike 常在手機上看）。

## 一個實驗的 README 五節

1. 想回答的問題（一句話） 2. 怎麼做（資料、方法、模型版本、日期、機器） 3. 數字（附區間）
4. 結論（一句話） 5. 沒做到的、下次想試的。第 4 節寫不出來就還沒完。

## 約定（家裡層已經有的，這裡只寫差異）

- log 照 `logging-conventions`；進入點呼叫 `setup_logging("rag_lab")`
- 收工更新 `HANDOFF.md`（`handoff` skill）；兩台機器以 GitHub 為真相，收工要 push
- 說「做完了」之前先跑 `superpowers:verification-before-completion`

## 寫程式之前，照這個順序問

1. 需要存在嗎？ 2. 專案裡已有？ 3. 標準庫做得到？ 4. 已裝套件做得到？ 5. 一行能寫完？ 6. 才寫最少的可動程式碼。
不砍的：log、例外處理、測試、邊界檢查。

## 不要做的

- 不在這裡寫實驗數字與歷史，那些放各實驗的 README 與 `docs/`。超過 60 行就搬東西出去。
