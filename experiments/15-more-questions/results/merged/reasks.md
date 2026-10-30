公報換句話問：80/160；與舊題對錯一致 0.918（隨機舊題 0.584，1120 對）
法規同一條／同一張表：56/80

| 實驗 | 模型 | 對 | 題數 | 差距 [95% 區間] | 判定 |
|---|---|---|---|---|---|
| 10 | gpt-oss:20b | +cnpre − fixed | 240 | +0.03 [-0.03, +0.08] | 分不出 |
| 10 | gpt-oss:20b | +rows − fixed | 240 | -0.03 [-0.06, +0.01] | 分不出 |
| 10 | gpt-oss:20b | full − -cnpre | 240 | +0.05 [-0.00, +0.10] | 分不出 |
| 10 | gpt-oss:20b | full − -rows | 240 | -0.03 [-0.06, -0.01] | 分得出 |
| 10 | gpt-oss:20b | oracle − full | 240 | -0.05 [-0.12, +0.02] | 分不出 |
| 10 | qwen3:8b | +cnpre − fixed | 240 | +0.02 [-0.04, +0.08] | 分不出 |
| 10 | qwen3:8b | +rows − fixed | 240 | -0.01 [-0.04, +0.01] | 分不出 |
| 10 | qwen3:8b | full − -cnpre | 240 | +0.04 [-0.01, +0.08] | 分不出 |
| 10 | qwen3:8b | full − -rows | 240 | -0.02 [-0.04, +0.00] | 分不出 |
| 10 | qwen3:8b | oracle − full | 240 | -0.08 [-0.14, -0.02] | 分得出 |
| 01 | bge-m3 | dense/article − dense/fixed269 | 47 | +0.04 [-0.06, +0.15] | 分不出 |
| 01 | bge-m3 | bm25/article − bm25/fixed1000 | 47 | +0.17 [+0.04, +0.30] | 分得出 |
| 01 | bge-m3 | bm25/article − bm25/fixed269 | 47 | +0.11 [-0.02, +0.23] | 分不出 |
| 01 | bge-m3 | dense/article − bm25/article | 47 | -0.02 [-0.15, +0.11] | 分不出 |
| 01 | bge-m3 | dense/table-flat-fixed − dense/table-row | 22 | -0.32 [-0.55, -0.09] | 分得出 |
| 01 | bge-m3 | dense/table-row − dense/table-whole | 22 | +0.04 [-0.09, +0.18] | 分不出 |
| 01 | bge-m3 | bm25/table-flat-fixed − bm25/table-row | 22 | -0.18 [-0.41, +0.00] | 分不出 |
| 01 | bge-m3 | bm25/table-flat-fixed − bm25/table-whole | 22 | -0.14 [-0.32, +0.04] | 分不出 |
| 01 | bge-m3 | bm25/table-row − bm25/table-whole | 22 | +0.04 [+0.00, +0.14] | 分不出 |
| 07 | bge-m3 | dense/article+context − dense/article | 57 | +0.04 [+0.00, +0.09] | 分不出 |
| 07 | bge-m3 | dense/article − dense/article-noprefix | 57 | +0.00 [-0.05, +0.05] | 分不出 |
| 07 | bge-m3 | dense/article − dense/small-to-big | 57 | +0.00 [-0.05, +0.05] | 分不出 |
| 07 | bge-m3 | hybrid/article − dense/article | 57 | +0.00 [-0.09, +0.09] | 分不出 |
| 07 | bge-m3 | dense/article − bm25/article | 57 | -0.02 [-0.12, +0.09] | 分不出 |
| 07 | bge-m3 | hybrid/small-to-big − bm25/small-to-big | 57 | +0.00 [-0.07, +0.07] | 分不出 |
