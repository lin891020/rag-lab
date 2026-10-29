# HANDOFF

最後更新：2026-10-29 13:40（mini）。

## 現在在哪一步

09-messy-real 做完：30 份天生就亂的公報法規修正案（國圖開放資料 gov.xml 分層抽，PDF 進 git），01–08 的 pipeline 不改直接上，160 題兩軌跑完、README 五節寫滿、第 4 節有「08 預測 vs 真資料」對照表，根 README 加了 09 那列。計畫經 Mike 核准（`~/.claude/plans/jazzy-riding-harbor.md`）。
下一步看第四節；打開 `experiments/09-messy-real/README.md` 第 4、5 節。

## 已驗證

- 09 語料：`build_corpus.py` seed 0 掃 233 筆湊齊 30 份（草案 15／令 15、地方 14、掃描 4、亂碼 1、旋轉附表 7），`corpus/manifest.json`、`corpus/health.csv`。
- 09 `extract.py` 三路線跑完：OCR 42 頁（glm-ocr，快取在 `data/09-messy-real/ocr`）、對照表 rows 在 15 份草案的 7 份找到三欄表。
- 09 `run_retrieval.py`：72 格 × 160 題，random .00–.01，向量 cache cosine 1.0000；base／article .00（160/160 切不出）、ocr／article-cn+pre dense hit@2000 .89 [.83, .93]；修法配對差距：中文條號 +.55、前言留塊 +.23、OCR +.11 分得出，fixed −.03、rows +.01、前綴 +.01 分不出（hit@5 +.04 分得出）。`attribute.py` 錯因在 `results/attribution.csv`。
- 09 `run_read.py`：160 題 × 5 條件 × 2 模型 = 1,620 次 141 分鐘 0 錯誤、漂移 19/20；base 等於閉卷（.05／.04，差距含 0），fixed .48／.42，all .71／.64，oracle .84／.82；fixed − base +.43／+.39、all − fixed +.23／+.21 都分得出。
- `uv run pytest tests experiments/0*/tests` 與 `pre-commit run --all-files`：見本次 commit 訊息的數字（2026-10-29 13:40 mini）。
- 01–08 見各自 README；HANDOFF-2026-10-28 的「已驗證」照舊。

## 只是寫了、還沒驗證

- 09 題庫 160 題全是我出、我核，`test_questions` 只驗 quote 在文字裡，沒驗 expect；Mike 的 15 題抽查名單在 09 README 第 5 節。05 的 25 題、03／04 標記、07 的 10 題也都還沒抽查。
- 09 閱讀軌的 all 條件沒含 OCR（rows 路線沒接 ocr 的頁），掃描／亂碼那 20 題在閱讀軌永遠沒正解；「OCR 對閱讀值多少」只有檢索軌的 +.11 可推。
- 09 的單位評分（06 的 judge）把「月／個月」「日／工作日」「人／名」與 gpt-oss 的空單位判錯，寬鬆分 oracle .89／.93；沒改評分，README 兩個數都報。
- 08 對照表的修法這次只修了三條（中文條號、前言留塊、OCR），無框線「x 位置當欄線」沒遇到也沒修。

## 下一步

1. Mike 抽查 09 的 15 題（名單在 09 README 第 5 節），看 question／quote／expect 跟 PDF 對不對。
2. 決定 10 做什麼：候選 (a) rows 疊在 ocr 上、三個修法在閱讀軌拆開量；(b) 單位別名補齊 + 中文數字題分開算；(c) 要點／附表自己的切法（10 份沒有「條」）。
3. hit@1000 跟 1,000 字塊的耦合要解（切 900 或預算 1,200），之後的實驗都受影響。
