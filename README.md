# rag-lab

個人的 RAG 實驗室：把文件餵給本機模型，量它答得準不準、吐得出固定格式嗎。產出是每個實驗一頁的紀錄，程式只是算數字的工具。設計文件在 `docs/superpowers/specs/`。

## 實驗

| 編號 | 問題 | 一句話結論 |
|---|---|---|
| [01-chunking](experiments/01-chunking/README.md) | 法規 PDF 用哪種切法檢索最準？ | 1000 字固定切換成約 270 字小塊，1000 字預算內命中率 .57 → .80；照條切跟同大小固定切分不出，贏的是「小」不是「條」 |

## 怎麼跑

```bash
uv sync
uv run pytest tests experiments/01-chunking
uv run python experiments/01-chunking/run.py     # 需要本機 ollama 有 bge-m3
```
