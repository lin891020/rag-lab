### 條文題（40 題）

| 切法 | 檢索 | hit@5 | hit@1000 | hit@2000 |
|---|---|---|---|---|
| article | bm25 | .85 [.72, .95] | .78 [.65, .90] | .82 [.70, .93] |
| article | dense | .85 [.72, .95] | .80 [.68, .93] | .85 [.72, .95] |
| article | hybrid | .85 [.72, .95] | .75 [.60, .88] | .85 [.72, .95] |
| article | random | .00 [.00, .01] | .00 [.00, .00] | .01 [.01, .01] |
| article+context | bm25 | .85 [.72, .95] | .75 [.60, .88] | .82 [.70, .93] |
| article+context | dense | .90 [.80, .97] | .82 [.70, .93] | .90 [.80, .97] |
| article+context | hybrid | .85 [.72, .95] | .80 [.68, .93] | .85 [.72, .95] |
| article+context | random | .00 [.00, .01] | .00 [.00, .00] | .01 [.00, .01] |
| article-noprefix | bm25 | .80 [.68, .93] | .78 [.62, .90] | .85 [.72, .95] |
| article-noprefix | dense | .85 [.72, .95] | .80 [.68, .93] | .88 [.78, .97] |
| article-noprefix | hybrid | .80 [.68, .93] | .78 [.65, .90] | .82 [.70, .93] |
| article-noprefix | random | .00 [.00, .01] | .00 [.00, .00] | .01 [.01, .01] |
| page | bm25 | .88 [.75, .97] | .50 [.35, .65] | .75 [.60, .88] |
| page | dense | .80 [.68, .93] | .53 [.38, .68] | .75 [.60, .88] |
| page | hybrid | .80 [.68, .93] | .62 [.47, .78] | .75 [.60, .88] |
| page | random | .02 [.01, .02] | .00 [.00, .00] | .01 [.01, .01] |
| small-to-big | bm25 | .82 [.70, .93] | .75 [.60, .88] | .82 [.70, .93] |
| small-to-big | dense | .85 [.72, .95] | .80 [.68, .90] | .85 [.72, .95] |
| small-to-big | hybrid | .88 [.78, .97] | .78 [.65, .90] | .80 [.68, .93] |
| small-to-big | random | .00 [.00, .01] | .00 [.00, .00] | .01 [.01, .01] |

配對差距（切法之間），dense，hit@1000：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| article − article+context | −.03 [−.07, +.00] | 分不出 |
| article − article-noprefix | +.00 [+.00, +.00] | 分不出 |
| article − page | +.28 [+.10, +.45] | 分得出 |
| article − small-to-big | +.00 [−.07, +.07] | 分不出 |
| article+context − article-noprefix | +.03 [+.00, +.07] | 分不出 |
| article+context − page | +.30 [+.12, +.47] | 分得出 |
| article+context − small-to-big | +.03 [+.00, +.07] | 分不出 |
| article-noprefix − page | +.28 [+.10, +.45] | 分得出 |
| article-noprefix − small-to-big | +.00 [−.07, +.07] | 分不出 |
| page − small-to-big | −.28 [−.45, −.10] | 分得出 |

配對差距（切法之間），bm25，hit@1000：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| article − article+context | +.03 [+.00, +.07] | 分不出 |
| article − article-noprefix | +.00 [−.07, +.07] | 分不出 |
| article − page | +.28 [+.10, +.45] | 分得出 |
| article − small-to-big | +.03 [+.00, +.07] | 分不出 |
| article+context − article-noprefix | −.03 [−.07, +.00] | 分不出 |
| article+context − page | +.25 [+.07, +.42] | 分得出 |
| article+context − small-to-big | +.00 [−.07, +.07] | 分不出 |
| article-noprefix − page | +.28 [+.10, +.45] | 分得出 |
| article-noprefix − small-to-big | +.03 [−.05, +.12] | 分不出 |
| page − small-to-big | −.25 [−.42, −.07] | 分得出 |

配對差距（切法之間），hybrid，hit@1000：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| article − article+context | −.05 [−.12, +.00] | 分不出 |
| article − article-noprefix | −.03 [−.07, +.00] | 分不出 |
| article − page | +.12 [−.05, +.30] | 分不出 |
| article − small-to-big | −.03 [−.07, +.00] | 分不出 |
| article+context − article-noprefix | +.03 [−.05, +.10] | 分不出 |
| article+context − page | +.17 [+.03, +.35] | 分得出 |
| article+context − small-to-big | +.03 [+.00, +.07] | 分不出 |
| article-noprefix − page | +.15 [+.00, +.33] | 分不出 |
| article-noprefix − small-to-big | +.00 [−.07, +.07] | 分不出 |
| page − small-to-big | −.15 [−.33, +.00] | 分不出 |

配對差距（檢索之間），hybrid−dense，hit@1000：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| small-to-big − small-to-big | −.03 [−.15, +.07] | 分不出 |
| article − article | −.05 [−.15, +.05] | 分不出 |
| article-noprefix − article-noprefix | −.03 [−.13, +.07] | 分不出 |
| page − page | +.10 [−.05, +.25] | 分不出 |
| article+context − article+context | −.03 [−.15, +.07] | 分不出 |

配對差距（檢索之間），hybrid−bm25，hit@1000：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| small-to-big − small-to-big | +.03 [−.05, +.10] | 分不出 |
| article − article | −.03 [−.12, +.05] | 分不出 |
| article-noprefix − article-noprefix | +.00 [−.10, +.10] | 分不出 |
| page − page | +.12 [−.03, +.28] | 分不出 |
| article+context − article+context | +.05 [−.05, +.15] | 分不出 |

配對差距（檢索之間），dense−bm25，hit@1000：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| small-to-big − small-to-big | +.05 [−.07, +.20] | 分不出 |
| article − article | +.03 [−.10, +.15] | 分不出 |
| article-noprefix − article-noprefix | +.03 [−.12, +.17] | 分不出 |
| page − page | +.03 [−.17, +.23] | 分不出 |
| article+context − article+context | +.07 [−.07, +.23] | 分不出 |
