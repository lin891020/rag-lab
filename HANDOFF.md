# HANDOFF

最後更新：2026-10-30 08:40（mini）。

## 現在在哪一步

15-more-questions（題數要多少才分得出）**程式、題庫、事前宣告都好了，等跑**：Mike 2026-10-30 核准計畫（`~/.claude/plans/jazzy-riding-harbor.md`，監察員 agent 審過 13 條、必改 6 條都併了）。`scratchpad/chain15.sh` 已在背景候著，等 `fix.log` 出現 FIXCHAIN DONE 才碰 ollama：07 run → 01 run（新舊三個題庫、新標籤規則）→ 15 `run_read.py`（8 條件 × 2 模型 × 160 = 2,560 次，每格前 3 題會 log 秒數）→ score（新）→ score（10 的舊 160 列用 15 的尺重算到 `results/rescored-old/`）→ power → merge → report；結束寫 CHAIN15 DONE 到 `scratchpad/chain15.log`。掛了手動照這順序跑（run_read 可續跑）。跑完：README 3–5 節（事前宣告表對照合併表；題庫統計：退件率 0、ocr 覆蓋 160/160、base 141、重複 39、條號開頭 0、法規題抽查 10/10、公報題抽查 20 題的結果）、根 README 一列、playbook 第 0 節題數那句、01／07／10 README 各加「15 加倍後」一句。
**跑的時候跑測試一律 `uv run pytest tests experiments/*/tests --deselect tests/test_ollama.py --deselect experiments/12-rerank/tests/test_rerankers.py`**（那兩檔會打 bge-m3／載重排器，把跑著的聊天模型踢掉）。
10-fix-ablation **做完**（README 五節齊，根 README 有列，playbook 的「（10 補）」四格填了）：4,160 次 0 錯誤、漂移 47/52（5 次全在 gpt-oss）。oracle 列跑時標籤還帶法規名，排在 `chain_fix.sh` 重跑後再改那一列。11-abstain **做完**（README 五節齊、根 README 有列、playbook 第 5 節加了一條）：3,200 次 0 錯誤、漂移 37/40。12 的 `run_rerank.py` 2026-10-30 03:21 自動接棒開跑（`scratchpad/rerank12.log`）。
12-rerank（重排序）**跑著**： `run_rerank.py`（兩格 × 五重排器，需 `uv sync --group rerank`）再 `run_read.py`（1,280 次），估到隔天上午；掛了手動跑，都可續跑。
13-points-tables（沒有「條」的文件怎麼切）語料 30 份、169 題、程式、README 1–2 節都好了；`scratchpad/chain13.sh` 在 12 跑完後接著跑 OCR（1 份 2 頁）→ `run_retrieval.py` → `attribute.py` → `run_read.py`（1,690 次 ≈ 2.5 小時）。掛了手動照這順序跑。
14-native-source（公報 PDF vs 法規系統 HTML／附件）語料 16 份、87 題（13／09 原題）、三路線文字、程式、README 1–2 節都好了；`scratchpad/chain14.sh` 在 13 跑完後接著跑 `run_retrieval.py` → `attribute.py` → `run_read.py`（1,044 次 ≈ 1.5 小時）。要重跑抽字用 `uv run --group office python experiments/14-native-source/extract.py`。
`docs/2026-10-29-real-data-playbook.md`（01–10 整理成一條流程）初稿已推，10–14 的數字跑完補。
**Mike 的要求（2026-10-29）：實驗都要完整跑、機器不要空著、不要反覆上下模型**——跑的時候不要再叫 ollama 做別的事（13 的 OCR 與向量因此排在 12 之後）。

## 已驗證

- 2026-10-30 09:00 01／07 審查決議實作並 commit（469fd21，趕在 chain_fix 跑 01 之前）：固定切標籤 ≥ min(100, 一半)、頁級標籤看本文、第一塊一定讀、run.py 旗標；01＋07 測試 33 個過。15 scaffold（35114ea）：全套測試（上面那條 deselect）exit 0；power.py 的 MDE 與監察員手算一致（+cnpre − fixed MDE@320 = .07）。
- 2026-10-30 03:21 11 閱讀軌 3,200 次跑完算分：P0 硬答 .91／.86，P1（schema 加 `found`）有依據可看時 .09／.11、有答案的題不掉；沒依據時照答 .66／.94；引句濾網兩層（在依據裡、含答的數字，2026-10-30 07:15 Mike 核准加的，重算不叫模型）：gpt-oss 編引句第一層擋九成、qwen3 抄真句答假數要第二層（.66 → .23）；誤擋 .04–.08。報表 `_f` 把 .997 印成「.00」的 bug 修了（14 個 report.py）。
- 2026-10-29 22:18 10 閱讀軌 4,160 次跑完算分：值錢的是前綴（+.17／+.16，幾乎全是哪一份題）與令文留塊（關掉 −.21／−.17，條號題 .56 → .17）；OCR +.07；一條一塊比 fixed-1000 只 +.05 區間含 0；對照表 ±0；全開 .82／.72，寬鬆 .82／.82；加一個與關一個排序一致。數字在 `experiments/10-fix-ablation/results/report.md`。
- 2026-10-29 19:20 三個獨立 agent 審了 01–10 的量尺程式（Mike：「前面的會不會也有這種問題 你找找」）。要動數字的：**09 的 oracle 標籤帶法規名**，「哪一份」題 31/33 等於看標籤答，oracle .84／.82 偏高（估掉到 .7 左右，與 all 持平）——09、10 的 oracle 重跑排在 `scratchpad/chain_fix.sh`（14 之後；drop oracle 列再續跑）；**07 頁級 −.28** 大半是「超過 1,000 字的頁就算排第一也判沒放進預算」（18% 的頁），改「第一塊一定讀」約 −.10，要重算；**01 fixed269 bm25 .83** 是標籤灌水（塊沾到條文一個字就算命中，兩成命中塊裡沒答案），要求塊含條文一半以上 → .70，dense 不變，要重算；**01 表格逐列切丟了 7 張表**（含 t15 正解），列偵測已修，chain_fix 重跑 01；**05 表五六第 0 頁表頭少一層**，qwen3「每頁一塊並排 −.13 分得出」扣掉那 6 題變分不出，chain_fix 重跑 48 次。不動數字的：09 attribute 的 over_budget 混了三種原因（無前綴格多半是預算用完，不是標頭）、preamble_dropped 誤標 1 題（已修）、article-cn 塊丟掉「第N條」行害 q061／q067 永遠不命中（.01）；09 README「base 是整份前 1,000 字」錯，其實是空依據；02／03／05／06 幾句措辭跟資料不符。**README 的數字與措辭等 chain_fix 跑完一次改，舊數字留著對照。**
- 2026-10-29 18:30 獨立 agent 審了 11／12／13／14 的量尺程式（Mike 問「需不需要另一個模型驗證」）：跑前修掉 4 個依據組法的 bug——11 的 same-doc 窗組不出來時原本會送閉卷提示（14 題）、same-doc 標籤帶法規名把哪一份題答案送上（15 題）；13 的 points 切塊把「一、」丟掉害 5 題永遠不算命中；14 的附件把公報 PDF 再附一次（去重後附件 43/51，仍有 34 題的 quote 出現 ≥2 次，來自總說明與對照表，README 要報每題 gold 塊數）。評分那邊也修了：拒答片語錨定、空單位／空法規名算棄答、parse_ok 進報表、real 列也做 leak。10 與 12 的 `article_chunks` 同樣丟掉「第N條」行（2 題），10 已在跑，不改，README 註記。
- 14：16 份原生頁抓到（14 份版本跟公報同日、1 份用法規資料庫歷史版、2 份只有現行版標記）、66 個附件（pdf 64、odt 2）；三路線文字抽完，quote 覆蓋 pdf 87/87、html 55/87、html+att 81/87；13 個測試綠；切塊乾跑三路線 × 三切法都有塊。
- 13：語料 30 份（要點 20／附表 10）體檢過；169 題 quote 全驗到（E2422248 掃描待 OCR）；15 題獨立 agent 抽查 15/15 對；17+ 測試綠；rows 路線 10 份找到 8 份的表。
- 12：11 個測試綠；煙霧測試 3 題 × 5 重排器（bge／Qwen3-Reranker 在 MPS 上跑得動、listwise 走 ollama）、漂移 6/6；新依賴 optional group `rerank` 裝好（torch 2.15 MPS）。
- 11：14 個測試綠（P0 逐字同 09、leak 抓中文／阿拉伯數字、濾網去空白比對）；煙霧測試 2 題 × 2 提示 × 5 依據 20 次 OK（P1 在沒答案的依據上 6/6 棄答）。
- 10 `extract.py` ocrrows 30 份；`score.py` 用新尺重算 09（`results/09-rescored/`）；`tests/test_run_read` 證明 10 組的依據跟 09 餵模型的字數相同；煙霧測試 2 題 × 3 條件 OK。
- 09 語料：`build_corpus.py` seed 0 掃 233 筆湊齊 30 份（草案 15／令 15、地方 14、掃描 4、亂碼 1、旋轉附表 7），`corpus/manifest.json`、`corpus/health.csv`。
- 09 `extract.py` 三路線跑完：OCR 42 頁（glm-ocr，快取在 `data/09-messy-real/ocr`）、對照表 rows 在 15 份草案的 7 份找到三欄表。
- 09 `run_retrieval.py`：72 格 × 160 題，random .00–.01，向量 cache cosine 1.0000；base／article .00（160/160 切不出）、ocr／article-cn+pre dense hit@2000 .89 [.83, .93]；修法配對差距：中文條號 +.55、前言留塊 +.23、OCR +.11 分得出，fixed −.03、rows +.01、前綴 +.01 分不出（hit@5 +.04 分得出）。`attribute.py` 錯因在 `results/attribution.csv`。
- 09 `run_read.py`：160 題 × 5 條件 × 2 模型 = 1,620 次 141 分鐘 0 錯誤、漂移 19/20；base 等於閉卷（.05／.04，差距含 0），fixed .48／.42，all .71／.64，oracle .84／.82；fixed − base +.43／+.39、all − fixed +.23／+.21 都分得出。
- `uv run pytest tests experiments/*/tests` 與 `pre-commit run --all-files`：見本次 commit 訊息的數字（2026-10-29 13:40 mini）。
- 09 題庫抽查 15 題（seed 9 名單）：Mike 太累改由一個沒看過題目的 agent 從頁面截圖獨立作答再比對，15/15 答案對、0 錯；它指出 q037 與 q144 的 quote 撐不滿答案（同句在別處先出現／只涵蓋 7 條裡 5 條）、q067 的附表條次要不要算沒寫明，答案本身都對，quote 沒改（改了會動檢索命中）。
- 01–08 見各自 README；HANDOFF-2026-10-28 的「已驗證」照舊。

## 只是寫了、還沒驗證

- 09 題庫剩下 145 題沒抽查（15 題已由獨立 agent 核過，見「已驗證」）。05 的 25 題、03／04 標記、07 的 10 題也都還沒抽查。
- 09 閱讀軌的 all 條件沒含 OCR（rows 路線沒接 ocr 的頁），掃描／亂碼那 20 題在閱讀軌永遠沒正解；「OCR 對閱讀值多少」只有檢索軌的 +.11 可推。
- 09 的單位評分（06 的 judge）把「月／個月」「日／工作日」「人／名」與 gpt-oss 的空單位判錯，寬鬆分 oracle .89／.93；沒改評分，README 兩個數都報。
- 08 對照表的修法這次只修了三條（中文條號、前言留塊、OCR），無框線「x 位置當欄線」沒遇到也沒修。

## 下一步

1. 等 12 跑完（rerank 1,600 列、read 1,280 列）→ 同上，playbook 第 4 節加重排。
2. 等 13 跑完（retrieval 72 格、read 1,690 列）→ 同上，playbook 第 3 節加「沒有條的文件」。
3. 等 14 跑完（retrieval 54 格、read 1,044 列）→ 同上，playbook 第 2 節加「拿得到原始格式就不要用 PDF」。之後候選見 playbook「還沒量過的」的四條（題庫加倍、30B 模型對照、換文件類型、重出題審查），再加文件改版、真人問法。
4. hit@1000 跟 1,000 字塊的耦合要解（切 900 或預算 1,200），之後的實驗都受影響。
