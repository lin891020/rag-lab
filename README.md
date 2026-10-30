# rag-lab

個人的 RAG 實驗室：把文件餵給本機模型，量它答得準不準、吐得出固定格式嗎。產出是每個實驗一頁的紀錄，程式只是算數字的工具。設計文件在 `docs/superpowers/specs/`；01–10 的結論整理成一條流程在 [docs/2026-10-29-real-data-playbook.md](docs/2026-10-29-real-data-playbook.md)。

## 實驗

| 編號 | 問題 | 一句話結論 |
|---|---|---|
| [01-chunking](experiments/01-chunking/README.md) | 法規 PDF 用哪種切法檢索最準？ | 1000 字固定切換成約 270 字小塊，1000 字預算內命中率 .57 → .80；照條切跟同大小固定切分不出，贏的是「小」不是「條」 |
| [02-json-output](experiments/02-json-output/README.md) | 提示詞拜託、json-mode、schema 約束，哪種 JSON 最可靠？ | 給了依據只要一個數字時，三種逼法輸出一模一樣：格式錯誤 0、值答對率 .88／.95 全同；錯的在讀表格，不在格式 |
| [03-table-format](experiments/03-table-format/README.md) | 表格餵純文字、markdown、HTML、逐列鍵值，哪種讀對格最多？ | markdown 與逐列鍵值都贏攤平文字（qwen3 .67 → .79／.84，gpt-oss .73 → .95／.87，區間不含 0）；HTML 沒幫助；範圍列與寬表仍是主要錯誤 |
| [04-table-lookup](experiments/04-table-lookup/README.md) | 範圍展開成上下限、只餵問到的列，救得了剩下的讀表錯嗎？ | 救不了：四個配對差距區間都含 0，合用在 gpt-oss 反而 −.09；真正的誘餌是表註裡的數字和寬表的欄 |
| [05-cross-page](experiments/05-cross-page/README.md) | 跨頁表格續頁沒表頭：逐頁讀、每頁一塊並排、規則接回，差多少？ | 損失只在「要靠欄名定位」的 6 題出現，接回表頭也沒救（那是 04 的範圍列問題）；分得出的是「每頁一塊並排」比只餵續頁差 −.13；偵測規則 4/4 真跨頁、108 鄰表 0 誤接 |
| [06-ocr-source](experiments/06-ocr-source/README.md) | 同一張表從 300 dpi 頁圖 OCR 回來，模型讀表掉多少？哪條本地 OCR 能用？ | glm-ocr（0.9B，提示尾加換行）跟文字層分不出（.73 vs .79、.88 vs .93）；提示照抄就掉 .10；Apple Vision 掉 .35–.43、tesseract chi_tra 等於閉卷 |
| [07-chunking-2](experiments/07-chunking-2/README.md) | 標題鏈前綴、LLM 脈絡句、頁級、父子、混合檢索，哪個讓條文找得更準？ | 一條一塊之後全都動不了：前綴 0、脈絡句 +.03、父子 0、混合 −.05～+.03（區間含 0）；頁級 −.28 分得出，輸在塞不進預算 |
| [08-messy](experiments/08-messy/README.md) | 把乾淨法規故意弄亂六種（頁首頁尾、兩欄、串接、手機拍、影印三代、無框線表），哪種弄壞哪一步、掉幾分？ | 只有兩種分得出：兩欄用 pdfplumber 抽把條號黏掉（一條一塊 −.80，固定切只 −.10）、表格沒框線用 text 策略欄黏成一格（gpt-oss −.11，攤平反而只 −.07）；頁首頁尾、串接、兩級掃描都在 ±.06 內分不出；附資料體檢工具 `doc_health.py` |
| [09-messy-real](experiments/09-messy-real/README.md) | 30 份天生就亂的公報法規修正案（對照表、掃描、亂碼、橫式附表），01–08 的 pipeline 不改直接上，08 的對照表準不準？ | 症狀中了三列、原因沒一列一樣（公報條號是中文數字同行，01 的切條 160/160 切不出）；沒預測到 cid 亂碼、前言被丟、要點沒條、塊撞預算四種；三個修法把檢索 .00 → .89、閱讀閉卷 → .71／.64，上限 .84／.82；rows 路線檢索看不出差、閱讀救回「抄到現行欄」 |
| [10-fix-ablation](experiments/10-fix-ablation/README.md) | 09 的三個修法一次開一個、一次關一個，各值多少？評分尺補單位後數字變多少？ | 值錢的是兩個「不丟字」：《法規名》前綴 +.17／+.16（幾乎全是「哪一份」題）、令文留塊關掉 −.21／−.17（條號題 .56 → .17）；OCR +.07；一條一塊比固定切只 +.05 區間含 0；對照表 ±0；全開 .82／.72，gpt-oss 嚴格分低 .10 全是空單位 |
| [11-abstain](experiments/11-abstain/README.md) | 依據裡沒答案時本機模型多常硬答？提示給棄答的路救回多少、有答案的題掉多少？程式驗引句擋掉多少？ | P0 硬答 .91／.86；schema 加 `found` 後有依據可看時降到 .09／.11，有答案的題 −.01／−.00 區間含 0；但沒依據時照答 .66／.94，要程式層擋；兩層引句濾網（在依據裡、含答的數字）擋 gpt-oss 編的引句九成、qwen3 抄真句答假數 .66 → .23，誤擋 .04–.08 |
| [12-rerank](experiments/12-rerank/README.md) | 第一階段 dense 之後接重排器，哪一種讓正解更常落在預算內、答對率加多少、一題幾秒？ | bge-reranker-v2-m3 值得：預算內有正解 +.11／+.21（一條一塊／固定切）、答對 +.05／+.06 分得出、3 秒一題；RRF ±.03；Qwen3 生成式重排 +.05 但 37 秒；LLM 列表式 ±0 到 −.14 且 31–40 秒 |

## 怎麼跑

```bash
uv sync
uv run pytest tests experiments/*/tests                       # 全部實驗
uv run python experiments/01-chunking/run.py     # 需要本機 ollama 有 bge-m3
uv run python experiments/02-json-output/run.py  # 需要 qwen3:8b、gpt-oss:20b；之後 score.py、report.py
uv run python experiments/03-table-format/run.py # 同上；語料已在 corpus/，重建要 01 的附件 PDF
uv run python experiments/04-table-lookup/run.py # 同上；讀 corpus/tables.json，不需要 PDF
uv run python experiments/08-messy/build_corpus.py  # 弄亂語料（要 01 的 law.pdf 與 glm-ocr）；之後 run_text.py、run_table.py、doc_health.py <資料夾>
uv run python experiments/09-messy-real/build_corpus.py  # 從國圖 gov.xml 抽 30 份（PDF 已在 corpus/）；之後 extract.py（掃描頁要 glm-ocr）、run_retrieval.py、attribute.py、run_read.py、score.py、report.py
```
