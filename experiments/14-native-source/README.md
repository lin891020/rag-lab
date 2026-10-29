# 14 拿得到乾淨來源時省多少：同一批公告，公報 PDF 來 vs 主管法規系統的 HTML／附件來

## 1. 想回答的問題

08–13 一直在跟 PDF 抽字的損失打仗（頁首頁尾、首字分行、cid 亂碼、旋轉附表、塊撞預算）。如果對方給得出原始資料，這些損失有多少本來就不用付？拿 09／13 裡 16 份行政院公報的公告，到各部會主管法規共用系統（或全國法規資料庫）抓同一份法規的 HTML 全文與附件，用**同一批題目**量：找得到、答得對各差多少；差的是 PDF 的哪種損失。

## 2. 怎麼做

**別人怎麼解的**：
- 搜「原生格式 vs PDF 對 RAG 的影響」：Docling／Unstructured／Tika 的比較全是廠商文；arXiv 2604.04948（葡萄牙行政文件）量的是 PDF 不同解析器（naive 86.9 → Docling 94.1 → 人工 Markdown 97.1），arXiv 2508.15817 量 PPTX→Markdown。共同說法是「保留結構的轉換比來源格式重要」，**沒有同一批文件兩種來源的對照**。14 補這個洞，小規模。
- Apache Tika：什麼格式都吃、只給文字。這裡的 `html` 路線（去標籤成文字）就是 Tika 式；Tika 本尊是 Java，沒裝。
- 來源調查在 `docs/research/2026-10-29-non-pdf-sources.md`：行政院公報 open data 的 HTML 只有令文、附件一律「請參見 PDF」；能配對的是各部會**主管法規共用系統**（同一家廠商：`LawContent.aspx?id=`，GET `LawResult.aspx?NLawTypeID=all&KW=…` 可搜）與全國法規資料庫（`LawAll.aspx?pcode=`，歷史版本 `LawOldVer.aspx`）。
- **本機**：13 的 `chunkers`（points、fixed）、`run_retrieval`／`run_read`／`score`／`attribute`；09／13 的題庫與 quote 驗證。

**語料**（`fetch_native.py`，`corpus/manifest.json`，抓取日 2026-10-29）：09／13 裡行政院公報與總統府公報、非預告的 18 份，找到 16 份（僑務委員會沒有法規系統、交通部 motclaw 憑證錯誤，2 份排除）：

| 來源 | 份數 | 系統 |
|---|---|---|
| 13 要點類 | 8 | 教育部 ×3、農業部 ×2、關務署（財政部系統）、金管會、經濟部 |
| 13 附表類 | 1 | 農業部（一覽表） |
| 09 令修正（辦法／標準） | 4 | 農業部、內政部、經濟部 ×2 |
| 09 法律／命令（全國法規資料庫） | 3 | 不動產估價師法、中小企業發展條例、保險業辦法 |

版本核對：法規系統放現行版本，14 份的「修正日期」就是公報那天；保險業辦法之後又修過，改抓法規資料庫的歷史版本（113-07-04）；**2 份（教育部終身學習要點、經濟部科技產業園區收費標準）2026 年又修過、系統上找不到歷史版本頁，照現行版本放進去並標 `same_version=false`**，數字另算。附件 66 個：`.pdf` 64（多半就是公報電子檔或附表 PDF）、`.odt` 2；沒有 `.doc`。

**三路線**（`extract.py`，`corpus/text/<sysid>.<route>.txt`）：

| 路線 | 做什麼 | quote 找得到（數字／條號／哪一份，共 59／8／20 題） |
|---|---|---|
| pdf | 13／09 的 `ocr` 路線文字（公報 PDF，原樣） | 59／8／20 |
| html | 法規系統頁面去標籤成文字，段落一行、表格「表頭：值｜…」一列一行、網站邊框去掉（Tika 式） | 41／0／14 |
| html+att | html 文字 + 附件原生轉文字（`.odt` odfpy、`.docx` python-docx、`.pdf` pypdf；`.doc` 記為拿不到） | 59／3／19 |

html 少的那些：條號題的 quote 在公報的**令文**（「修正第X點」），法規系統全文沒有令；附表類與附件類的內容在附件不在頁面（E2421778、E2428884 頁面只有 140／193 字）；數字題有 18 題的句子只在附表裡。這本身就是第一個發現：**法規系統的 HTML 是「本文」，附表仍是 PDF**。

**題庫**：13／09 那 16 份的題原封不動（`questions.jsonl`，87 題：數字 59、條號 8、哪一份 20；13 的 48 題、09 的 39 題，帶 `from`／`src_id`）。

**切塊與格**：fixed-1000、fixed-300、points（13 的）× 前綴 × dense／bm25／random；hit@5、hit@1000、hit@2000、預算內有正解；每格對同切法 pdf 路線的配對差距（`diffs.csv` 的 `pdf/same`）。閱讀軌：closed-book、oracle（quote 窗，pdf 文字找不到就 html+att）、pdf／fixed-300、html／fixed-300、pdf／points＋前綴、html+att／points＋前綴 × qwen3:8b `think:false`／gpt-oss:20b `think:"low"` × 87 題 = 1,044 次；P0 提示、10 的尺。

**五條規矩**：雜訊底線 random／closed-book；盲測基線 pdf 路線（13 的 pipeline）；bootstrap 95% + 配對差距；版本表（下）；漂移探針。

| 項目 | 值 |
|---|---|
| 日期／機器 | 2026-10-29／mini.local |
| 法規系統抓取日 | 2026-10-29 |
| ollama／bge-m3 | 0.40.2／`790764642607` |
| qwen3:8b／gpt-oss:20b digest | `500a1f067a9f`／`f38aa0c53da5` |
| odfpy／python-docx | （`uv sync --group office`）1.4.x／1.2.0 |

**預期**（事先宣告）：quote 找得到的那些題，html 比 pdf 預算內有正解 +.05–.10、答對率 +.05 以內分不出；html 路線在條號題與附表題是 0（來源沒那段）；html+att 跟 pdf 分不出（附件就是同一份 PDF）。

## 3. 數字

（跑完填）

## 4. 結論

（跑完填）

## 5. 沒做到的、下次想試的

（跑完填）
