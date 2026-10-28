# rag-lab

個人的 RAG 實驗室：把文件餵給本機模型，量它答得準不準、吐得出固定格式嗎。產出是每個實驗一頁的紀錄，程式只是算數字的工具。設計文件在 `docs/superpowers/specs/`。

## 實驗

| 編號 | 問題 | 一句話結論 |
|---|---|---|
| [01-chunking](experiments/01-chunking/README.md) | 法規 PDF 用哪種切法檢索最準？ | 1000 字固定切換成約 270 字小塊，1000 字預算內命中率 .57 → .80；照條切跟同大小固定切分不出，贏的是「小」不是「條」 |
| [02-json-output](experiments/02-json-output/README.md) | 提示詞拜託、json-mode、schema 約束，哪種 JSON 最可靠？ | 給了依據只要一個數字時，三種逼法輸出一模一樣：格式錯誤 0、值答對率 .88／.95 全同；錯的在讀表格，不在格式 |
| [03-table-format](experiments/03-table-format/README.md) | 表格餵純文字、markdown、HTML、逐列鍵值，哪種讀對格最多？ | markdown 與逐列鍵值都贏攤平文字（qwen3 .67 → .79／.84，gpt-oss .73 → .95／.87，區間不含 0）；HTML 沒幫助；範圍列與寬表仍是主要錯誤 |
| [04-table-lookup](experiments/04-table-lookup/README.md) | 範圍展開成上下限、只餵問到的列，救得了剩下的讀表錯嗎？ | 救不了：四個配對差距區間都含 0，合用在 gpt-oss 反而 −.09；真正的誘餌是表註裡的數字和寬表的欄 |
| [05-cross-page](experiments/05-cross-page/README.md) | 跨頁表格續頁沒表頭：逐頁讀、每頁一塊並排、規則接回，差多少？ | 損失只在「要靠欄名定位」的 6 題出現，接回表頭也沒救（那是 04 的範圍列問題）；分得出的是「每頁一塊並排」比只餵續頁差 −.13；偵測規則 4/4 真跨頁、108 鄰表 0 誤接 |

## 怎麼跑

```bash
uv sync
uv run pytest tests experiments/01-chunking experiments/02-json-output experiments/03-table-format experiments/04-table-lookup
uv run python experiments/01-chunking/run.py     # 需要本機 ollama 有 bge-m3
uv run python experiments/02-json-output/run.py  # 需要 qwen3:8b、gpt-oss:20b；之後 score.py、report.py
uv run python experiments/03-table-format/run.py # 同上；語料已在 corpus/，重建要 01 的附件 PDF
uv run python experiments/04-table-lookup/run.py # 同上；讀 corpus/tables.json，不需要 PDF
```
