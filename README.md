# rag-lab

個人的 RAG 實驗室：把文件餵給本機模型，量它答得準不準、吐得出固定格式嗎。產出是每個實驗一頁的紀錄，程式只是算數字的工具。01–10 的結論整理成一條流程在 [docs/2026-10-29-real-data-playbook.md](docs/2026-10-29-real-data-playbook.md)。換成雲端模型（Claude API）答題時的做法在 [docs/2026-11-04-hosted-model-playbook.md](docs/2026-11-04-hosted-model-playbook.md)。

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
| [13-points-tables](experiments/13-points-tables/README.md) | 沒有「條」的要點與附表怎麼切？照「一、二、」切、附表一列一塊，比固定切多多少？ | 固定切 300 字就夠：預算內有正解比 1,000 字 +.20；按點切跟 300 字分不出（+.02）；附表一列一塊檢索 +.02、閱讀 +.15／+.17 碰邊；答對率分得出的只有《法規名》前綴（哪一份題 +.46） |
| [14-native-source](experiments/14-native-source/README.md) | 拿得到法規系統的 HTML／附件時，比公報 PDF 省多少？ | 省不到：HTML 少了令文、附表、標題，32/87 題答案不在 HTML，答對 −.24；加附件補回後跟 PDF 打平；兩邊都有答案的題分不出——損失在內容不全不在格式，原生格式當補充不當替代 |
| [15-more-questions](experiments/15-more-questions/README.md) | 01／07／10 那些「分不出」是真的沒差還是題太少？題庫加倍重算 | 大多是真的差很小：25 對裡排除重複題後只有 2 對變分得出（bm25 一條一塊 vs 固定 1,000 字 +.17、表格一列一塊 vs 攤平 +.32）；對照表收斂成 ±.03 內；新題一半是舊題換句話問（對錯一致 92%），加倍實際約 1.5 倍 |
| [16-rule-lookup-compute](experiments/16-rule-lookup-compute/README.md) | 給一個負載，查規則與表格列、算出一個數：閉卷／加知識庫／加計算工具各掉在哪一步？ | 輸贏全在撈不撈得到對的那一列：直接給列答對 .88／.68，靠檢索 .28／.22（正解列全進依據只有四成）；計算工具 +.00～+.04，算術本來沒錯，錯在規則讀錯（停在中間值、沒分段累進）與撈錯列；qwen3 依據不全時會選小一級（.17–.27） |
| [17-table-retrieval](experiments/17-table-retrieval/README.md) | 16 的檢索瓶頸，換「先找表、整張給」與「先拆題再找」各救回多少？ | 最大一步是補回清表時丟掉的副標（60／75／90℃ 三張同名表分得開，答對兩模型都 +.15）；整張給只對 gpt-oss 有用（+.11），qwen3 撈對了表卻讀錯欄；拆題讓正解列全進 .60 → .67、答對沒升；離 oracle 仍差 .29／.41；gpt-oss 同設定重跑答對就晃 .04 |
| [18-table-kv](experiments/18-table-kv/README.md) | 17 的整張表改成每格帶欄名（內容不變），qwen3 讀錯欄會不會消失？ | 讀錯欄 8 → 3 次，但答對沒升（qwen3 −.02、gpt-oss +.00，都分不出）：錯誤換成讀錯列，「以下」範圍表最明顯；gpt-oss 自己重跑 cards 答案 149/160 同、答對 −.03，17 的 +.11 縮成 +.08 仍分得出 |
| [19-lookup-tool](experiments/19-lookup-tool/README.md) | 模型只說查哪張表、哪一列、哪一欄，程式回那一格，C1／C4／C5 答對比整張表給它讀多多少？ | 讀格子幾乎不再錯（查到時答對 .86／1.0），但總分沒升：qwen3 +.08 分不出、gpt-oss −.13 分得出；錯搬到「選哪幾列」——分段只查一段、「以下」讀成下面那段，而且看不到整張表的上下文。第一次跑另有提示例子被照抄、括號比對太嚴兩個我造成的錯，修掉後重跑 |
| [20-api-ceiling](experiments/20-api-ceiling/README.md) | 最強的模型（Claude Opus 5.5）用整張表、查表工具、正解列，C1／C4／C5 各答對多少？它也會點錯格嗎？ | 不會：查表工具 75 題全對、正解格一格沒漏（小模型選錯列 19–24 題）；整張表 .99，比 gpt-oss +.19 分得出。它在 3 題因題目電壓寫錯而棄答，引出題庫更正：5 題字跟法規不合，16–20 拿掉重算，判定都沒翻面。花 US$11 |
| [21-lookup-segment](experiments/21-lookup-segment/README.md) | 查表工具從回一格改成回整欄（同一份第一步查詢），小模型答對多多少？ | gpt-oss +.13 分得出（C4 +.30），兩次跑一樣；qwen3 +.03 分不出——C4 正解格全看得到了只對 4/23，因為它把分段負載乘一個係數。看得到不等於會算 |
| [22-degraded-scan](experiments/22-degraded-scan/README.md) | 06 的表做成品質差的掃描（兩級），glm-ocr 讀回來，小模型答對率掉多少？ | 沒掉：最差一級（150 dpi、歪 1.5°、模糊、雜訊、JPEG q30）L0 − L2 qwen3 −.02、gpt-oss +.01，分不出；表格裡的格子一格都沒多讀錯，掉分仍是「整張表回純文字」，L2 反而少了。Apple Vision 一樣差（格子 .54–.59）；Opus 讀圖格子 1.00，花 US$1.3 |
| [23-ocr-size](experiments/23-ocr-size/README.md) | 送 glm-ocr 前先把乾淨頁縮小，「整張表回純文字」會不會消失？ | 不會穩定消失：縮到一半救回 表九二、救不回 表四○，答對 +.02／+.01 分不出；縮到 1/3 有 6 張回純文字、答對掉 .10。ollama 自己會把大圖縮到上限，進模型一樣大的三級仍有一張表時有時無——回不回表格是個不穩的決定，不是圖太大 |
| [24-ocr-header](experiments/24-ocr-header/README.md) | glm-ocr 讀回的表，把多層表頭照文字層攤成一列，小模型答對率會不會追上文字層？ | 不會：qwen3 −.01、gpt-oss +.00，分不出。qwen3 在攤平後跟文字層一字不差的寬表救回 3 題取錯欄，別的表卻少對 4 題（2 題單位寫錯）；跟文字層的 .06 差距還在，剩下主要是表回純文字和 OCR 少了表的標題 |
| [25-hosted-tables](experiments/25-hosted-tables/README.md) | 換成雲端模型（Claude Haiku／Opus 5.5），表格頁給文字層、OCR 文字、乾淨頁圖、爛掃描，各答對多少、花多少？ | 全部到頂、分不出：Opus 四種都 82/82，Haiku .99–1.00；22–24 的 OCR 問題在雲端模型上消失。直接給頁圖不必 OCR，每題貴 3～4 倍（Haiku US$0.0007、Opus US$0.024）。這批題對雲端模型太簡單，要更難的題才分得出 |
| [26-hosted-retrieval](experiments/26-hosted-retrieval/README.md) | 換成雲端模型做「查規則再計算」160 題，依據給閉卷、三種檢索、正解列，檢索還是不是最大的損失？ | 是：oracle − cards Opus +.32、Haiku +.24，分得出；答對率幾乎等於檢索撈全率——正解列撈全時 Opus 221/223 對、沒撈全 3/158，有依據時答出口的 342 次全對，錯的都變成「找不到」。Haiku 在 cards 跟 Opus 同分、便宜 36 倍；Opus 閉卷憑記憶 .40，但答錯的也很有自信 |
| [27-hosted-agentic-search](experiments/27-hosted-agentic-search/README.md) | 給雲端模型一支搜尋工具（跟 26 的 cards 同一個索引），讓它自己下查詢、最多查 5 次，正解列撈全率和答對率能推多高？ | 推到頂：撈全率 .77 → Opus 1.0、Haiku .93；答對 Opus 127/127、Haiku .98／.96，比最好的固定流程 +.24、分得出。平均只查 1.6–1.9 次，每題貴約 2 倍（Opus US$0.038）。Haiku 多了一種錯：表格空白格不棄答、拿鄰欄推值 |
| [28-hosted-bigger-corpus](experiments/28-hosted-bigger-corpus/README.md) | 索引從一部規則擴成六部相近的法規（字數約 2 倍、條號全撞），雲端模型自己查的答對率掉多少？ | 沒掉：Opus 仍 127/127；Haiku .98 → .96，−.02 [−.06, +.03] 在它自己重跑的雜訊內。別部法規 Opus 21 題、Haiku 約 28 題有查到，但答錯的引句全出自原本那部；查詢次數、花費幾乎不變（Opus +2%）。連固定檢索撈全率都沒掉，表示這批干擾還太遠 |
| [29-amendment-diff](experiments/29-amendment-diff/README.md) | 同一部規則的新舊兩版（114 年全案修正、條號全重排），題目附新版某一條全文、工具只查舊版：雲端模型答得出它在舊版是第幾條、文字有沒有改、數值改成多少嗎？錯在找不到還是讀錯？ | 找對應幾乎不失手：撈全 Haiku .99、Opus 1.0，答對 .94／.95；有沒有改、數值改多少跟直接給舊條文一樣準（oracle − agent 兩個模型都 ±.02、分不出）。19 題次的錯一半是官方對照表的血緣跟內容相似不一致（文字從甲條搬來、血緣記在乙條；全部重寫仍寫移列），另一半是新增的條被硬配相近的舊條。T3 只有 12 題；US$7.00 |

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
