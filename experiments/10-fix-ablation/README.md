# 10 把 09 的三個修法拆開量，順手把評分尺修準

## 1. 想回答的問題

09 的三個修法（中文條號＋前言留塊、OCR、對照表一條一筆）在閱讀軌是綁在一起量的（all = .71／.64），而且 all 沒把 OCR 接進來。一次只開一個、或一次只關一個，每個修法對答對率各值多少？評分尺補上公報單位、加一個「單位空白不算錯」的寬鬆分之後，09 的數字變多少？中文數字跟阿拉伯數字的題答對率有差嗎？

## 2. 怎麼做

**別人怎麼解的**：
- **拆開量的兩種標準做法**：add-one（從底線一次加一個：CDF-RAG，arXiv 2504.12560）與 drop-one（從全開一次關一個：COLIEE 2026 的 cross-encoder 貢獻，arXiv 2609.26237；CodaRAG，arXiv 2604.10426）。綜述（arXiv 2506.00054）說兩種都常見，修法之間有交互作用時兩張表會不一致 → 兩張都報，不一致就是發現。
- **數字答案的評分慣例**：GAIA 的 scorer 去掉 $、%、逗號只比數字；SLMJury（arXiv 2606.07810）去單位字、展開 million／billion、相對誤差 1e-9；ORCA（arXiv 2511.02589）單位換算後數字相等就算對。**沒有人把「單位字串不一樣」算錯**。09 沿用 06 的嚴格尺（數字對且單位對），比慣例嚴 → 這裡補同義單位、另加寬鬆分對齊慣例；嚴格分留著看模型守不守 schema。
- **中文數字**：NUMCoT（ACL 2024，arXiv 2406.02864）中文數字↔阿拉伯數字轉換比英文差、數字越大越差；arXiv 2501.04927 中英數字翻譯「萬／億」最弱，建議提示寫明換算規則或事後用 cn2an 修。→ 這裡只量不修，「提示加換算規則」放第 5 節。
- **本機**：09 的 `extract.rows_text`／`ocr_text`（疊成 ocrrows 只差一個 `pages` 參數）、`run_read.Evidence`／`budget_evidence`（一字不改，`tests/test_run_read.py` 拿 09 的 `raw.jsonl` 證明依據相同）、`score.UNIT_ALIASES`／`judge_value`。

**語料與題庫**：不複製，讀 `../09-messy-real/corpus/text/*.{base,ocr,rows}.txt`、`../09-messy-real/questions.jsonl`（160 題：數字 109、條號 18、哪一份 33）、manifest。只新產一條路線 `corpus/text/<sysid>.ocrrows.txt`（`extract.py`：先換 OCR 頁、再換對照表頁；09 語料裡有 OCR 頁的 5 份對照表 0 筆、有對照表的 7 份 OCR 0 頁，不重疊，測試證明 ocrrows 在前者等於 ocr、後者等於 rows）。向量 cache 共用 09 的。

**條件**（`prompts.CONDITIONS`，路線／切塊／前綴；依據組法、1,000 字預算、prompt、schema 全照 09）：

| 組 | 條件 | 路線 | 切塊 | 前綴 |
|---|---|---|---|---|
| 底線 | closed-book | — | — | — |
| 上限 | oracle | quote 前後 1,000 字 | — | — |
| 加一個（起點 fixed） | fixed | base | fixed-1000 | 無 |
| | +ocr | ocr | fixed-1000 | 無 |
| | +rows | rows | fixed-1000 | 無 |
| | +prefix | base | fixed-1000 | 有 |
| | +cnpre | base | article-cn+pre | 無 |
| 關一個（起點 full） | full | ocrrows | article-cn+pre | 有 |
| | −ocr（＝09 的 all） | rows | article-cn+pre | 有 |
| | −rows | ocr | article-cn+pre | 有 |
| | −prefix | ocrrows | article-cn+pre | 無 |
| | −pre | ocrrows | article-cn | 有 |
| | −cnpre（換成固定切） | ocrrows | fixed-1000 | 有 |

加一個的起點不是 09 的 base（一條一塊 .00，加什麼都還是 .00），是 08 的保底 fixed-1000。13 條件 × 2 模型（qwen3:8b `think:false`、gpt-oss:20b `think:"low"`、temperature 0、seed 0）× 160 題 = 4,160 次；09 的四格不沿用，全部重跑。

**評分**（`score.py`，09 的複本加三件事）：
1. `UNIT_ALIASES` 補公報單位，只收同義：元（新臺幣元）、日（天）、工作日（個工作日）、個月（月）、人（名、位）、次（回）、條、點、節、公分、公尺、公噸（噸）、ppm…。「日」不等於「工作日」、「月」不等於「年」，測試列了不該相等的對。
2. `value_lenient`：數字對、且單位對**或空字串**（gpt-oss 問「第幾條」「幾人」時常填空）。單位寫錯（「萬元」對「元」）還是錯。
3. 數字題標寫法：quote 裡的正解是阿拉伯數字（`arabic`）、中文數字（`cn`）、含「萬」的中文數字（`wan`），各 17／83／9 題。

**報表**：加一個（每個修法 − fixed）、關一個（full − 關掉它）、階梯（fixed − closed-book、full − fixed、oracle − full）各自配對 bootstrap 95%，嚴格分與寬鬆分並列；每條件按題型、按數字寫法；依據含正解比例。

**五條規矩**：雜訊底線 closed-book；盲測基線 fixed（08 的保底）；bootstrap 95% + 配對差距；版本表（下）；漂移探針每格重打 2 題。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-29／mini.local |
| ollama／bge-m3／glm-ocr | 0.40.2／`790764642607`／`89d735dd9c07`（OCR 文字沿用 09 的快取） |
| qwen3:8b／gpt-oss:20b digest | （跑完填） |
| Python／pypdf／pdfplumber／PyMuPDF | 3.12.15／6.21.3／0.11.10／2.0 |

## 3. 數字

### 09 用新尺重算（`results/09-rescored/`，09 的 `raw.jsonl` 一字不動）

| 模型 | 09 的條件 | 09 的尺 | 新尺嚴格 | 新尺寬鬆 |
|---|---|---|---|---|
| qwen3:8b | oracle | .84 | .87 [.82, .92] | .87 |
| qwen3:8b | all | .71 | .73 [.66, .80] | .73 |
| qwen3:8b | fixed | .48 | .50 [.42, .57] | .50 |
| gpt-oss:20b | oracle | .82 | .83 [.78, .89] | **.93 [.88, .96]** |
| gpt-oss:20b | all | .64 | .64 [.57, .72] | .75 [.69, .81] |
| gpt-oss:20b | fixed | .42 | .44 [.36, .51] | .52 [.44, .59] |

同義單位（個月／月、人／名）救回 qwen3 .02–.03；gpt-oss 的差距幾乎全在空單位（寬鬆分 +.10），它數字對、單位不填。兩個模型的上限用寬鬆尺看是 .87／.93，差距反過來了。

數字題按寫法，09 的 oracle（嚴格分）：

| 模型 | 中文數字 83 | 含「萬」9 | 阿拉伯數字 17 |
|---|---|---|---|
| qwen3:8b | .94 [.88, .99] | .89 [.67, 1.0] | **.71 [.47, .88]** |
| gpt-oss:20b | .83 [.75, .92]（寬鬆 .99） | .78 [.44, 1.0] | **.71 [.47, .94]** |

跟 NUMCoT 的預期相反：中文數字沒有比較差，**阿拉伯數字的題反而最差**。看題目：17 題阿拉伯數字幾乎全是附表（收費表、節數表）裡的格子，差的是「表攤平後數字對不回列」（09 第 3 節的 q021／q022／q026），不是數字寫法。含「萬」的 9 題區間太寬，分不出。

### 閱讀軌（跑完填）

## 4. 結論

（跑完填）

## 5. 沒做到的、下次想試的

（跑完填）
