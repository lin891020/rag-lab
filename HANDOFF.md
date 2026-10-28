# HANDOFF

最後更新：2026-10-29（mini，深夜自動實驗中）。

## 現在在哪一步

Mike 2026-10-28 23:24 交代「自己去做更多實驗：混合來源、跨頁沒表頭、不同格式怎麼切與傳遞，先查別人怎麼做」，早上驗收。
做了：四份調查筆記（`docs/research/`）、05 跨頁表格（做完、推了）、06 掃描來源（語料與結構層做完，模型格子 820 次正在跑）。
GitHub：`lin891020/rag-lab`（公開）。這兩個實驗**沒有先經 Mike 核准計畫**（他授權了「看你去怎麼設計」），README 第 2 節各有「別人怎麼解的」。

## 已驗證

- `uv run pytest tests experiments/01-chunking experiments/02-json-output experiments/03-table-format experiments/04-table-lookup experiments/05-cross-page experiments/06-ocr-source`：全綠（2026-10-29 00:15 mini，05 37 題、06 29 題）；`pre-commit run --all-files` 綠（00:00 時）。
- 05 `detect.py`：112 對，4/4 真跨頁接上、108 鄰表負例 0 誤接。
- 05 `run.py` 304 次 22 分鐘、漂移 14/16；數字在 05 README 第 3 節：page .87/.92、pages .74/.79、merged .79/.87；pages − page −.13 兩個模型都分得出。
- 06 `structure.py`：格子召回 glm .57、vision .71、tesseract .15；glm 8 張表只回純文字。
- 01–04 見各自 README；HANDOFF-2026-10-28 的「已驗證」照舊。

## 只是寫了、還沒驗證

- **06 `run.py` 正在跑**（00:14 開始，5 條件 × 2 模型 × 82 題），結果、README 第 3／4 節、repo README 的 06 列都還沒寫。跑完要 `score.py` → `report.py`。
- 05 題庫 38 題：25 題新出（c01–c25、k07–k13），正解對 PDF 原圖標，**Mike 沒抽查**；名單 c11、c16、c19、c23、k08。
- 03 的 n03、n17、n39、n44、n53 與 04 的 row 標記 t01、n54、n17、n28、n26 也還沒抽查。
- 06 的 glm-ocr 只試了一種提示（`Table Recognition:`）；8 張回純文字的表沒試第二種提示。

## 下一步

1. 06 跑完：`score.py`、`report.py`、README 第 3／4 節、repo README、commit、push。
2. Mike 抽查 05 的 5 題、03／04 的名單。
3. 下一個：（a）06 續集「退化掃描 + glm 第二種提示 + 欄重建」，或（b）01 續集「切分：標題鏈前綴、元素切分、表＋註綁一起」（`docs/research/2026-10-29-chunking-survey.md` 的建議順序），或（c）混合格式來源（DOCX／XLSX 同一份法規，`parsing-toolchains` 筆記路 B）。
