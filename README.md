# rag-lab

個人的 RAG 實驗室：把法規 PDF 餵給模型，量它找得到、答得準嗎。29 個實驗，每個一個資料夾、一頁 README（問題、做法、數字附區間、結論、沒做到的），程式只是算數字的工具。

**2026-11-08 收尾。** 這條線（一部法規、自己出的題）到此為止；再開的條件是出現一份真的要從裡面找答案的文件、而且有人會說答對沒。

## 結論

1. **本機小模型（qwen3:8b、gpt-oss:20b，01–24）：答錯幾乎都不是格式問題，是「沒拿到對的那一段」或「表格讀錯格」。** 檢索這端最值錢的是不丟字（《法規名》前綴、令文留塊），比任何切法技巧都大；塊切小、接重排器各有一份。表格那端換格式只是把錯搬家：讀錯欄變讀錯列、讀錯列變不會算，上限是 8B 模型的推理，不是餵法。
2. **雲端模型（Claude Haiku／Opus 5.5，25–29）：讀什麼都到頂，唯一的瓶頸是檢索。** 表格、OCR、頁圖、爛掃描分不出差；正解撈全就幾乎全對、沒撈全就答不出。給它一支搜尋工具自己查就到頂，語料加倍、新舊版比對都不掉。Haiku 跟 Opus 差 .00–.02、便宜約 30 倍。上線配方：Haiku ＋ 搜尋工具 ＋ 程式擋硬答。
3. **方法。** 每個實驗都有雜訊底線、bootstrap 區間、盲測基線、版本表、漂移探針，跑前寫下預期、跑完記哪些沒中。題庫加倍只讓 25 對「分不出」裡的 2 對變分得出，那些「沒差」大多是真的沒差。

**限制**：全部是一部法規加自己出的題，沒有真實使用者的問題。能說的是「技術上做得到、知道怎麼量、知道怎麼配」，不能說「在真場景驗過」。
**教訓**：先用最強的模型量上限，再決定要不要在小模型上花力氣。事後看，18、19、21–24 與 28、29 這八個實驗帶來的資訊很少，約四分之一可以省。

做法整理成兩頁：本機小模型版 [docs/2026-10-29-real-data-playbook.md](docs/2026-10-29-real-data-playbook.md)、雲端模型版 [docs/2026-11-04-hosted-model-playbook.md](docs/2026-11-04-hosted-model-playbook.md)。開工前的文獻與工具調查在 [docs/research/](docs/research/README.md)。

## 實驗

數字與區間在各實驗 README 第 3 節，這裡只留一句話。

### 文件怎麼切、怎麼讀（本機小模型，01–15）

| 編號 | 問題 | 一句話結論 |
|---|---|---|
| [01-chunking](experiments/01-chunking/README.md) | 法規 PDF 用哪種切法檢索最準？ | 小塊贏：1000 字切成約 270 字，命中率 .57 → .80；照條切沒有額外好處 |
| [02-json-output](experiments/02-json-output/README.md) | 拜託、json-mode、schema，哪種 JSON 最可靠？ | 三種逼法輸出一模一樣；錯在讀表格，不在格式 |
| [03-table-format](experiments/03-table-format/README.md) | 表格餵純文字、markdown、HTML、逐列鍵值，哪種讀對最多？ | markdown 與逐列鍵值都贏攤平文字；HTML 沒幫助 |
| [04-table-lookup](experiments/04-table-lookup/README.md) | 範圍展開、只餵問到的列，救得了讀表錯嗎？ | 救不了；誘餌是表註的數字和寬表的欄 |
| [05-cross-page](experiments/05-cross-page/README.md) | 跨頁表格續頁沒表頭，怎麼餵差多少？ | 只在要靠欄名定位的題掉分；每頁一塊並排最差（−.13） |
| [06-ocr-source](experiments/06-ocr-source/README.md) | 300 dpi 頁圖 OCR 回來掉多少？哪條本地 OCR 能用？ | glm-ocr 跟文字層分不出；Apple Vision 掉 .35–.43、tesseract 等於閉卷 |
| [07-chunking-2](experiments/07-chunking-2/README.md) | 標題前綴、脈絡句、頁級、父子、混合檢索，哪個更準？ | 一條一塊之後全動不了；頁級 −.28 |
| [08-messy](experiments/08-messy/README.md) | 乾淨法規故意弄亂六種，哪種弄壞哪一步？ | 只有兩種傷：兩欄排版黏掉條號、無框線表欄黏成一格 |
| [09-messy-real](experiments/09-messy-real/README.md) | 30 份天生就亂的公報，pipeline 不改直接上會怎樣？ | 08 猜中症狀、沒猜中原因；三個修法把檢索 .00 → .89 |
| [10-fix-ablation](experiments/10-fix-ablation/README.md) | 09 的三個修法各值多少？ | 值錢的是不丟字：《法規名》前綴 +.17、令文留塊 +.21；OCR +.07 |
| [11-abstain](experiments/11-abstain/README.md) | 依據裡沒答案時小模型多常硬答？救得回多少？ | 九成硬答；schema 加 found 降到 .1，沒依據時仍要程式擋引句 |
| [12-rerank](experiments/12-rerank/README.md) | dense 之後接重排器，哪一種值得？ | bge-reranker-v2-m3 值得：答對 +.05／+.06、3 秒一題；LLM 重排沒用還慢 |
| [13-points-tables](experiments/13-points-tables/README.md) | 沒有「條」的要點與附表怎麼切？ | 固定切 300 字就夠，按點切分不出；真分得出的只有《法規名》前綴 |
| [14-native-source](experiments/14-native-source/README.md) | 拿得到法規系統的 HTML 時，比公報 PDF 省多少？ | 省不到：HTML 少了令文與附表，答對 −.24；當補充不當替代 |
| [15-more-questions](experiments/15-more-questions/README.md) | 那些「分不出」是真沒差還是題太少？題庫加倍重算 | 25 對裡只有 2 對變分得出；多數差距真的很小 |

### 查表接計算、OCR 表格（本機小模型，16–24；20 用 Opus 量上限）

| 編號 | 問題 | 一句話結論 |
|---|---|---|
| [16-rule-lookup-compute](experiments/16-rule-lookup-compute/README.md) | 給一個負載，查規則與表格列、算出一個數，掉在哪一步？ | 輸贏全在撈不撈得到對的那一列；計算工具幫不上 |
| [17-table-retrieval](experiments/17-table-retrieval/README.md) | 16 的檢索瓶頸，先找表整張給、先拆題，各救回多少？ | 補回表的副標是最大一步（+.15）；拆題讓撈全升、答對沒升 |
| [18-table-kv](experiments/18-table-kv/README.md) | 整張表改成每格帶欄名，讀錯欄會消失嗎？ | 讀錯欄少了，答對沒升：錯換成讀錯列 |
| [19-lookup-tool](experiments/19-lookup-tool/README.md) | 模型只說查哪格、程式回那一格，比整張表讀多對多少？ | 讀格子幾乎不錯了，總分沒升：錯搬到選哪幾列 |
| [20-api-ceiling](experiments/20-api-ceiling/README.md) | 最強的模型（Opus 5.5）用同樣的表與工具，答對多少？ | 查表 75 題全對、一格沒點錯；順手抓出題庫 5 題寫錯 |
| [21-lookup-segment](experiments/21-lookup-segment/README.md) | 查表工具從回一格改成回整欄，小模型多對多少？ | gpt-oss +.13 分得出；qwen3 看得到也不會算 |
| [22-degraded-scan](experiments/22-degraded-scan/README.md) | 品質差的掃描經 glm-ocr 讀回來，小模型掉多少？ | 沒掉；掉分仍是整張表回純文字 |
| [23-ocr-size](experiments/23-ocr-size/README.md) | 送 OCR 前先縮圖，「表回純文字」會消失嗎？ | 不會穩定消失；縮到 1/3 反而掉 .10 |
| [24-ocr-header](experiments/24-ocr-header/README.md) | OCR 表的多層表頭攤成一列，會追上文字層嗎？ | 不會；差距剩表回純文字與少了標題 |

### 換雲端模型（25–29）

| 編號 | 問題 | 一句話結論 |
|---|---|---|
| [25-hosted-tables](experiments/25-hosted-tables/README.md) | 換雲端模型，表格頁給文字層、OCR、頁圖、爛掃描各答對多少？ | 全到頂、分不出；頁圖不必 OCR，每題貴 3–4 倍 |
| [26-hosted-retrieval](experiments/26-hosted-retrieval/README.md) | 雲端模型做「查規則再計算」160 題，檢索還是最大的損失嗎？ | 是：正解撈全就幾乎全對、沒撈全就答不出；Haiku 同分、便宜 36 倍 |
| [27-hosted-agentic-search](experiments/27-hosted-agentic-search/README.md) | 給雲端模型一支搜尋工具自己查，能推多高？ | 推到頂：Opus 撈全 1.0、127/127 全對；每題貴約 2 倍 |
| [28-hosted-bigger-corpus](experiments/28-hosted-bigger-corpus/README.md) | 索引擴成六部相近法規，答對掉多少？ | 沒掉；這批干擾還太遠 |
| [29-amendment-diff](experiments/29-amendment-diff/README.md) | 新舊兩版條號全重排，只給舊版的搜尋工具，找得到對應、看得出改了什麼嗎？ | 幾乎不失手（.94／.95）；剩下的錯是對照表血緣跟內容不一致、新增的條被硬配舊條 |

## 怎麼跑

```bash
uv sync
uv run pytest tests experiments/*/tests        # 全部測試
uv run python experiments/01-chunking/run.py   # 每個實驗自己帶程式：run.py 或 run_*.py；17–29 另有 chain*.sh 串「跑 → 評分 → 報告」
```

- 本機模型走 host 的 ollama `127.0.0.1:11434`：向量 `bge-m3`、聊天 `qwen3:8b`／`gpt-oss:20b`、OCR `glm-ocr`。
- 20、22、25–29 要雲端模型：`.env` 放 `ANTHROPIC_API_KEY`（不進 git）。
- 小資料（法規 PDF、題庫、結果 CSV）在 git；嵌入快取、大 PDF、原始回覆在 `data/`，不進 git。每個實驗要哪些，看它 README 第 2 節。
