# HANDOFF

最後更新：2026-10-29 09:30（mini）。

## 現在在哪一步

08-messy 做完：把 01 的乾淨法規故意弄亂六種（頁首頁尾、兩欄、串接另一份法規、手機拍、影印三代、無框線表），兩軌跑完、README 五節寫滿、第 4 節有「亂 → 壞在哪一步 → 掉幾分 → 體檢抓得到？ → 修法」對照表，根 README 加了 08 那列。計畫經 Mike 核准（`~/.claude/plans/jazzy-riding-harbor.md`）。
附帶一個資料體檢工具 `experiments/08-messy/doc_health.py <資料夾>`，真資料來了先丟給它。
下一步看第四節；打開 `experiments/08-messy/README.md` 第 4 節。

## 已驗證

- 08 `run_text.py`：40 題 × 5 抽字組合 × 2 切法 × 3 檢索器，漂移 cosine 1.0000；頁首頁尾、兩欄／pypdf、串接 article 都 .78–.80 分不出；兩欄／pdfplumber 解析到 1 條、article .00（−.80 分得出）、fixed-1000 −.08／−.10；串接 fixed／dense −.10 分得出。
- 08 `run_table.py`：82 題 × 7 條件 × 2 模型 = 1,148 次 69 分鐘 0 錯誤、漂移 23/28（差的全是 gpt-oss t01）；clean .79／.93；三級掃描 −.05～−.06 分不出；無框線 text 策略 −.09／−.11（gpt-oss 分得出）、攤平 −.02／−.07。
- 08 `structure.py`：格子召回 scan-clean .87、phone .90、copy3 .95（表九二／表四○ 退化後翻成 HTML）、borderless-text .55、flat 0。
- 08 `doc_health.py` 在 23 個 PDF（`results/health.csv`）：乾淨法規 0 誤報、頁首頁尾抓到 2 行、兩欄 156/157 頁、掃描 0 文字層頁、無框線 13/17 標對齊文字。
- `uv run pytest tests experiments/0*/tests` 全綠、`pre-commit run --all-files` 綠（2026-10-29 09:30 mini，數字見 commit）。
- 01–07 見各自 README；HANDOFF-2026-10-28 的「已驗證」照舊。

## 只是寫了、還沒驗證

- 08 的對照表「修它要什麼」那欄是推論，沒有一條真的修過再量（無框線用 x 位置當欄線、頁首頁尾清掉）。
- 08 兩級掃描是人工退化（Pillow），對 glm-ocr 不夠髒；真手機照沒拍。
- 05 題庫 25 題新出、03／04 標記、07 保留 10 題：Mike 都還沒抽查（名單在 HANDOFF-2026-10-28 與 05 README）。

## 下一步

1. Mike 看 08 README 第 4 節的對照表，決定真資料來之前要不要先修哪一列（最便宜的是無框線「x 位置當欄線」，一步接 pdfplumber `explicit_vertical_lines`）。
2. Mike 的第 ③ 項已查完：`docs/research/2026-10-29-messy-zhtw-sources.md`，第一候選是行政院公報的法規修正對照表（開放資料 XML 整批抓）；等 Mike 決定領域範圍再開 09 的計畫。
3. 題庫擴到 150 題（01、07、08 都卡在 ±.09 分不出）。
