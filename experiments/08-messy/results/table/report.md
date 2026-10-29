### 全部（82 題）

| 模型 | 條件 | parse_ok | schema_ok | value_ok | value_only_ok | eval_count | seconds |
|---|---|---|---|---|---|---|---|
| gpt-oss:20b | borderless-flat | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .85 [.77, .93] | .02 [.00, .06] | 89 | 4.0 |
| gpt-oss:20b | borderless-text | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .82 [.73, .90] | .00 [.00, .00] | 94 | 4.4 |
| gpt-oss:20b | clean | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .93 [.87, .98] | .02 [.00, .06] | 103 | 4.6 |
| gpt-oss:20b | closed-book | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .06 [.01, .12] | .00 [.00, .00] | 117 | 4.9 |
| gpt-oss:20b | scan-clean | .99 [.96, 1.0] | .99 [.96, 1.0] | .88 [.80, .94] | .00 [.00, .00] | 93 | 4.3 |
| gpt-oss:20b | scan-copy3 | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .88 [.80, .94] | .00 [.00, .00] | 81 | 3.7 |
| gpt-oss:20b | scan-phone | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .87 [.79, .93] | .00 [.00, .00] | 80 | 3.7 |
| qwen3:8b | borderless-flat | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .77 [.67, .85] | .00 [.00, .00] | 36 | 2.6 |
| qwen3:8b | borderless-text | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .71 [.60, .80] | .00 [.00, .00] | 40 | 2.8 |
| qwen3:8b | clean | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .79 [.70, .88] | .00 [.00, .00] | 41 | 2.9 |
| qwen3:8b | closed-book | .99 [.96, 1.0] | .99 [.96, 1.0] | .05 [.01, .10] | .00 [.00, .00] | 66 | 4.0 |
| qwen3:8b | scan-clean | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .73 [.63, .83] | .00 [.00, .00] | 42 | 2.9 |
| qwen3:8b | scan-copy3 | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .79 [.71, .88] | .00 [.00, .00] | 45 | 2.9 |
| qwen3:8b | scan-phone | 1.0 [1.0, 1.0] | 1.0 [1.0, 1.0] | .74 [.65, .83] | .00 [.00, .00] | 44 | 2.8 |

配對差距，qwen3:8b，value_ok：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| clean − scan-clean | +.06 [−.04, +.16] | 分不出 |
| clean − scan-phone | +.05 [−.05, +.15] | 分不出 |
| clean − scan-copy3 | +.00 [−.09, +.09] | 分不出 |
| clean − borderless-text | +.09 [−.02, +.18] | 分不出 |
| clean − borderless-flat | +.02 [−.06, +.11] | 分不出 |
| scan-clean − scan-phone | −.01 [−.06, +.04] | 分不出 |
| scan-clean − scan-copy3 | −.06 [−.13, +.01] | 分不出 |
| scan-clean − borderless-text | +.02 [−.10, +.15] | 分不出 |
| scan-clean − borderless-flat | −.04 [−.15, +.07] | 分不出 |
| scan-phone − scan-copy3 | −.05 [−.11, +.00] | 分不出 |
| scan-phone − borderless-text | +.04 [−.09, +.16] | 分不出 |
| scan-phone − borderless-flat | −.02 [−.15, +.10] | 分不出 |
| scan-copy3 − borderless-text | +.09 [−.01, +.20] | 分不出 |
| scan-copy3 − borderless-flat | +.02 [−.07, +.12] | 分不出 |
| borderless-text − borderless-flat | −.06 [−.15, +.02] | 分不出 |

配對差距，gpt-oss:20b，value_ok：

| 配對差距 | 差距 [95% 區間] | 判定 |
|---|---|---|
| clean − scan-clean | +.05 [−.02, +.12] | 分不出 |
| clean − scan-phone | +.06 [−.02, +.13] | 分不出 |
| clean − scan-copy3 | +.05 [−.01, +.11] | 分不出 |
| clean − borderless-text | +.11 [+.04, +.20] | 分得出 |
| clean − borderless-flat | +.07 [+.01, +.15] | 分得出 |
| scan-clean − scan-phone | +.01 [−.02, +.06] | 分不出 |
| scan-clean − scan-copy3 | +.00 [−.04, +.04] | 分不出 |
| scan-clean − borderless-text | +.06 [−.02, +.15] | 分不出 |
| scan-clean − borderless-flat | +.02 [−.07, +.12] | 分不出 |
| scan-phone − scan-copy3 | −.01 [−.05, +.02] | 分不出 |
| scan-phone − borderless-text | +.05 [−.05, +.15] | 分不出 |
| scan-phone − borderless-flat | +.01 [−.09, +.11] | 分不出 |
| scan-copy3 − borderless-text | +.06 [−.02, +.15] | 分不出 |
| scan-copy3 − borderless-flat | +.02 [−.06, +.11] | 分不出 |
| borderless-text − borderless-flat | −.04 [−.12, +.05] | 分不出 |
