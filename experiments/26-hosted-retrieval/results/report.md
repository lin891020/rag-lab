# 26 report

送出的模型：claude-haiku-5-5, claude-opus-5-5；anthropic 2.3.0；呼叫 2400 次、失敗 0、被停 0；花費 US$13.8052

## 答對（嚴格，有答案的題；defect 不計）

| 模型 | closed | struct@1500 | cards | decomp@4000 | oracle |
|---|---|---|---|---|---|
| claude-haiku-5-5 | .00 [.00, .00] | .37 [.29, .45] | .61 [.52, .69] | .74 [.66, .81] | .85 [.79, .91] |
| claude-opus-5-5 | .40 [.32, .49] | .39 [.32, .47] | .61 [.52, .69] | .76 [.69, .83] | .93 [.88, .97] |
| qwen3:8b（17，同一把尺重算） | — | .23 | .33 | .27 | .69 |
| gpt-oss:20b（17，同一把尺重算） | — | .24 | .55 | .60 | .87 |

有答案的題 127；qwen3／gpt-oss 沒跑閉卷。

## 配對差距（答對嚴格；按題 bootstrap，括號是按 cluster）

粗體是跑前宣告的主檢定，只有它們下判定；其餘只描述。

| 配對 | 模型 | 差距 [按題] | [按 cluster] | 判定 |
|---|---|---|---|---|
| **oracle − cards** | claude-haiku-5-5 | +0.24 [+0.16, +0.33] | [+0.15, +0.32] | 分得出 |
| **oracle − cards** | claude-opus-5-5 | +0.32 [+0.24, +0.40] | [+0.27, +0.38] | 分得出 |
| **cards − struct@1500** | claude-haiku-5-5 | +0.24 [+0.17, +0.31] | [+0.20, +0.28] | 分得出 |
| **cards − struct@1500** | claude-opus-5-5 | +0.21 [+0.15, +0.28] | [+0.18, +0.26] | 分得出 |
| decomp@4000 − cards | claude-haiku-5-5 | +0.13 [+0.06, +0.20] | [+0.06, +0.21] | （描述） |
| decomp@4000 − cards | claude-opus-5-5 | +0.16 [+0.09, +0.22] | [+0.10, +0.22] | （描述） |
| struct@1500 − closed | claude-haiku-5-5 | +0.37 [+0.29, +0.45] | [+0.32, +0.42] | （描述） |
| struct@1500 − closed | claude-opus-5-5 | -0.01 [-0.08, +0.06] | [-0.07, +0.07] | （描述） |
| oracle − decomp@4000 | claude-haiku-5-5 | +0.11 [+0.02, +0.20] | [+0.00, +0.21] | （描述） |
| oracle − decomp@4000 | claude-opus-5-5 | +0.17 [+0.09, +0.24] | [+0.08, +0.25] | （描述） |
| claude-opus-5-5 oracle − claude-haiku-5-5 oracle | claude-opus-5-5 | +0.08 [+0.02, +0.13] | [+0.02, +0.16] | （描述） |
| claude-opus-5-5 cards − claude-haiku-5-5 cards | claude-opus-5-5 | +0.00 [-0.03, +0.03] | [-0.03, +0.04] | （描述） |
| closed-2 − closed | claude-haiku-5-5 | +0.00 [+0.00, +0.00] | [+0.00, +0.00] | （描述） |
| struct@1500-2 − struct@1500 | claude-haiku-5-5 | +0.02 [+0.00, +0.04] | [+0.00, +0.04] | （描述） |
| cards-2 − cards | claude-haiku-5-5 | +0.01 [-0.02, +0.03] | [-0.02, +0.03] | （描述） |
| decomp@4000-2 − decomp@4000 | claude-haiku-5-5 | +0.01 [+0.00, +0.02] | [+0.00, +0.02] | （描述） |
| oracle-2 − oracle | claude-haiku-5-5 | +0.01 [-0.03, +0.05] | [-0.02, +0.04] | （描述） |

## 按鏈答對（嚴格，有答案的題）

| 模型 | 條件 | C1 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | closed | .00 | .00 | .00 | .00 | .00 |
| claude-haiku-5-5 | struct@1500 | .04 | .00 | .00 | .91 | .96 |
| claude-haiku-5-5 | cards | .96 | .04 | .08 | 1.0 | 1.0 |
| claude-haiku-5-5 | decomp@4000 | .96 | .46 | .31 | 1.0 | 1.0 |
| claude-haiku-5-5 | oracle | 1.0 | 1.0 | .42 | .83 | 1.0 |
| claude-opus-5-5 | closed | .04 | .00 | .35 | .83 | .85 |
| claude-opus-5-5 | struct@1500 | .04 | .00 | .00 | 1.0 | 1.0 |
| claude-opus-5-5 | cards | .96 | .04 | .08 | 1.0 | 1.0 |
| claude-opus-5-5 | decomp@4000 | .96 | .46 | .42 | 1.0 | 1.0 |
| claude-opus-5-5 | oracle | 1.0 | 1.0 | .65 | 1.0 | 1.0 |
| qwen3:8b | struct@1500 | .04 | .00 | .00 | .39 | .73 |
| qwen3:8b | cards | .50 | .04 | .08 | .30 | .73 |
| qwen3:8b | decomp@4000 | .27 | .08 | .12 | .26 | .62 |
| qwen3:8b | oracle | 1.0 | .85 | .08 | .52 | 1.0 |
| gpt-oss:20b | struct@1500 | .04 | .00 | .00 | .57 | .65 |
| gpt-oss:20b | cards | .96 | .00 | .23 | .61 | .96 |
| gpt-oss:20b | decomp@4000 | .88 | .31 | .31 | .61 | .88 |
| gpt-oss:20b | oracle | 1.0 | .81 | .65 | .91 | 1.0 |

## 正解列全進依據時的答對（有答案的題）

| 模型 | 條件 | 全進：對／題 | 不全：對／題 |
|---|---|---|---|
| claude-haiku-5-5 | struct@1500 | 47/50 | 0/77 |
| claude-haiku-5-5 | cards | 75/75 | 2/52 |
| claude-haiku-5-5 | decomp@4000 | 93/98 | 1/29 |
| claude-opus-5-5 | struct@1500 | 50/50 | 0/77 |
| claude-opus-5-5 | cards | 75/75 | 2/52 |
| claude-opus-5-5 | decomp@4000 | 96/98 | 1/29 |
| qwen3:8b | struct@1500 | 29/50 | 0/77 |
| qwen3:8b | cards | 39/75 | 3/52 |
| qwen3:8b | decomp@4000 | 32/98 | 2/29 |
| gpt-oss:20b | struct@1500 | 31/50 | 0/77 |
| gpt-oss:20b | cards | 64/75 | 6/52 |
| gpt-oss:20b | decomp@4000 | 70/98 | 6/29 |

## 該棄答的題、選小一級（critical）

| 模型 | 條件 | 該棄答有棄答 | 選小一級（選型題） |
|---|---|---|---|
| claude-haiku-5-5 | closed | 1.0（28） | .00（52） |
| claude-haiku-5-5 | struct@1500 | 1.0（28） | .00（52） |
| claude-haiku-5-5 | cards | 1.0（28） | .04（52） |
| claude-haiku-5-5 | decomp@4000 | 1.0（28） | .00（52） |
| claude-haiku-5-5 | oracle | 1.0（28） | .06（52） |
| claude-opus-5-5 | closed | .86（28） | .10（52） |
| claude-opus-5-5 | struct@1500 | .96（28） | .00（52） |
| claude-opus-5-5 | cards | 1.0（28） | .00（52） |
| claude-opus-5-5 | decomp@4000 | 1.0（28） | .00（52） |
| claude-opus-5-5 | oracle | .96（28） | .00（52） |
| qwen3:8b | struct@1500 | .43（28） | .23（52） |
| qwen3:8b | cards | .00（28） | .21（52） |
| qwen3:8b | decomp@4000 | .00（28） | .21（52） |
| qwen3:8b | oracle | .11（28） | .02（52） |
| gpt-oss:20b | struct@1500 | .79（28） | .02（52） |
| gpt-oss:20b | cards | .46（28） | .17（52） |
| gpt-oss:20b | decomp@4000 | .54（28） | .04（52） |
| gpt-oss:20b | oracle | .71（28） | .00（52） |

## 雜訊

| 比較 | 條件 | 同答 | 答對 這次 → 另一次 |
|---|---|---|---|
| Haiku 兩次 | closed | 155/155 | 28 → 28 |
| Haiku 兩次 | struct@1500 | 153/155 | 75 → 77 |
| Haiku 兩次 | cards | 152/155 | 105 → 106 |
| Haiku 兩次 | decomp@4000 | 153/155 | 122 → 123 |
| Haiku 兩次 | oracle | 147/155 | 136 → 137 |
| claude-haiku-5-5 對 20（C1／C4／C5） | cards | 91/91 | — |
| claude-haiku-5-5 對 20（C1／C4／C5） | oracle | 91/91 | — |
| claude-opus-5-5 對 20（C1／C4／C5） | cards | 91/91 | — |
| claude-opus-5-5 對 20（C1／C4／C5） | oracle | 89/91 | — |

## 花費（含失敗的呼叫；Haiku 兩次合計）

| 模型 | 條件 | 呼叫 | US$ | 每次呼叫 US$ | 輸入 token（平均） | 輸出 token（平均） |
|---|---|---|---|---|---|---|
| claude-haiku-5-5 | closed | 320 | 0.07 | 0.0002 | 1011 | 234 |
| claude-haiku-5-5 | struct@1500 | 320 | 0.13 | 0.0004 | 2563 | 293 |
| claude-haiku-5-5 | cards | 320 | 0.17 | 0.0005 | 3775 | 318 |
| claude-haiku-5-5 | decomp@4000 | 320 | 0.22 | 0.0007 | 4807 | 401 |
| claude-haiku-5-5 | oracle | 320 | 0.13 | 0.0004 | 1709 | 445 |
| claude-opus-5-5 | closed | 160 | 2.59 | 0.0162 | 1011 | 607 |
| claude-opus-5-5 | struct@1500 | 160 | 2.08 | 0.0130 | 2563 | 138 |
| claude-opus-5-5 | cards | 160 | 2.89 | 0.0180 | 3775 | 147 |
| claude-opus-5-5 | decomp@4000 | 160 | 3.58 | 0.0224 | 4807 | 158 |
| claude-opus-5-5 | oracle | 160 | 1.95 | 0.0122 | 1709 | 268 |
