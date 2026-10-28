# HANDOFF

最後更新：2026-10-29 01:45（mini，深夜自主實驗收工）。

## 現在在哪一步

Mike 2026-10-28 23:24 交代「自己去做更多實驗：混合來源、跨頁沒表頭、不同格式怎麼切與傳遞，先查別人怎麼做」，早上驗收。
做了：四份調查筆記（`docs/research/`）、05 跨頁表格、06 掃描來源，都做完、README 五節寫滿、推了。早上給 Mike 的一頁在 `docs/2026-10-29-morning-report.md`。
GitHub：`lin891020/rag-lab`（公開）。這兩個實驗**沒有先經 Mike 核准計畫**（他授權了「看你去怎麼設計」），README 第 2 節各有「別人怎麼解的」。

## 已驗證

- `uv run pytest tests experiments/01-chunking … experiments/06-ocr-source`：全綠（2026-10-29 01:45 mini）；`pre-commit run --all-files` 綠。04／05 兩個測試原本在整套一起跑時會拿到別實驗的模組（函式內 import），改成頂層 import。
- 05 `detect.py`：112 對，4/4 真跨頁接上、108 鄰表負例 0 誤接。
- 05 `run.py` 304 次 22 分鐘、漂移 14/16；數字在 05 README 第 3 節：page .87/.92、pages .74/.79、merged .79/.87；pages − page −.13 兩個模型都分得出。
- 06 `structure.py`：格子召回 glm2 .87、glm .57、vision .71、tesseract .15。06 `run.py` 984 次（820 + glm2 164）69 分鐘、漂移 18/24（差的全是 gpt-oss t01／t02a）；數字在 06 README 第 3 節：clean − glm2 分不出（+.06／+.05）、glm2 − glm +.10／+.11 分得出、vision 掉 .35–.43、tesseract 等於閉卷。
- 01–04 見各自 README；HANDOFF-2026-10-28 的「已驗證」照舊。

## 只是寫了、還沒驗證

- 05 題庫 38 題：25 題新出（c01–c25、k07–k13），正解對 PDF 原圖標，**Mike 沒抽查**；名單 c11、c16、c19、c23、k08。
- 03 的 n03、n17、n39、n44、n53 與 04 的 row 標記 t01、n54、n17、n28、n26 也還沒抽查。
- 06 的 glm2 路線是跑完 glm 後才加的（提示尾加換行），同晚同機補跑；表九二、表四○ 兩個提示都只回純文字，沒試第三種。

## 下一步

1. Mike 看 `docs/2026-10-29-morning-report.md`，抽查 05 的 5 題（c11、c16、c19、c23、k08）與 03／04 的名單。
2. 下一個：（a）06 續集「退化掃描 + glm 第二種提示 + 欄重建」，或（b）01 續集「切分：標題鏈前綴、元素切分、表＋註綁一起」（`docs/research/2026-10-29-chunking-survey.md` 的建議順序），或（c）混合格式來源（DOCX／XLSX 同一份法規，`parsing-toolchains` 筆記路 B）。
