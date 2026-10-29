# 調查：PDF 以外的繁中公開文件來源（給 14 用）

日期 2026-10-29。問題：01–13 的資料全是 PDF。真實系統會拿到 Word／Excel／ODT／HTML，而且「拿得到原始格式」時 PDF 抽字的那些損失（08–13 量的）本來就不用付。要量這件事，得找到**同一份文件同時有 PDF 與原始格式**的來源。

## 結論（一句話）

台灣公部門可編輯文件的法定格式是 ODF（ODT／ODS），不是 Word；但真正能批次拿到的「同一份文件的非 PDF 版」是各部會**主管法規共用系統**的 HTML 全文（要點每一點一列）加附件（DOC／ODT／PDF），而且有歷史版本。行政院公報的 open data 只有令文是 HTML，附件一律「請參見 PDF」。

## 查到的來源

| 來源 | 格式 | 內容 | 能不能批次 | 跟 09／13 對得上嗎 |
|---|---|---|---|---|
| 行政院公報 open data（`gazette.nat.gov.tw/egFront/OpenData/downloadXML.jsp`、`download.jsp`） | XML（每筆 `HTMLContent`、`GazetteHTML`）+ 每筆 PDF | `HTMLContent` 只有令文（主旨、生效日），要點本文與附表寫「（請參見PDF）」；`GazetteHTML` 連到 `Eg.htm`，同樣只有令文，旁邊只有 `Eg01.pdf` | 當日可；歷史日期的參數試了 `date`／`publishDate`／`d`／`year-month-date` 都回今天的，舊站 `/old/OpenData` 已轉址到新站 | 文件對得上（同一則公告），但非 PDF 的部分只有令文 |
| **各部會主管法規共用系統**（同一家廠商：`edu.law.moe.gov.tw`、`law.moa.gov.tw`、`glrs.moi.gov.tw`、`mojlaw.moj.gov.tw`…，頁面 `LawContent.aspx?id=FLxxxxxx`） | HTML 全文；附件 `Download.ashx?FileID=…` 多為 `.doc`／`.odt`／`.pdf`；另有「歷史法規」「法規沿革」 | 要點全文一點一段（含款目）、附表多半是附件 | 搜尋頁是 JS／postback，用法規名在 web 搜尋或各部會站內查得到；一份一份找 | **13 的 16 份行政院公報要點、09 的中央令修正**都該有對應頁；預告草案不會有（還沒發布） |
| 臺北市法規查詢系統（`laws.gov.taipei/Law/LawSearch/LawFileList/FLxxxxxx`，`/Law/File/00000xxxxx`） | 附件 ODT（真的 ODT，7 KB 起）、PDF | 法規條文是 HTML；附件多是表單、附表 | 列表頁 JS 產生（curl 只拿到 112 bytes），附件檔案直接抓得到；沒找到索引或 open data | 09／13 的臺北市公報項目（gov.xml 裡 182 筆要點／附表／收費）可逐一對 |
| 全國法規資料庫 open data（data.gov.tw 18289 等） | XML／ZIP 全量 | 法律與中央命令的條文（結構化：條、項） | 可 | 09 的中央令修正（辦法、規則、標準）對得上；地方自治法規不在裡面 |
| data.gov.tw 一般資料集 | ODS／XLSX／CSV 很多 | 統計表、清冊（資料表，不是文件） | 可 | 另一種題（試算表格子查詢），跟 03／04 的表格題接得上 |
| 新北、桃園法規查詢系統首頁 | 只看到 PDF 連結 | — | 未深查 | — |

本機沒裝 LibreOffice（`soffice` 不在），所以「ODT 轉 PDF 再走 pypdf」這條路線要裝才有；`odfpy` 也沒裝。

## 搜了什麼、沒找到

- 全國法規資料庫的 Word 下載：沒有，只有 XML／ZIP。
- data.gov.tw 以 ODT／DOCX 提供要點全文的資料集：沒找到；找到的 ODT／DOC 都是各部會法規頁的附件（表單）。
- 行政院公報全文的 ODT／DOC：沒有；公報電子版只有 PDF 與令文 HTML。
- 立法院提案、考選部試題、央行新聞稿的 DOCX 批次：沒找到可批次的來源。

## 對 14 的建議

拿 13 的 16 份行政院公報要點（必要時加 09 的中央令修正），到各部會法規共用系統找同一份要點的 HTML 全文與附件，用**同一套題目**量：公報 PDF（13 的 pipeline）vs HTML 結構切（一點一塊，不用猜條號規則）vs HTML 攤平（Tika 式只留文字）vs 附件 DOC／ODT 原生解析。先確認每一份法規共用系統上的「修正日期」就是公報那次（之後再修過的要排除或標記），因為那裡放的是現行版本。順便，「歷史法規」是之後做「文件改版」實驗的現成來源。

## 連結

- 行政院公報 open data：<https://data.gov.tw/dataset/5959>、<https://gazette.nat.gov.tw/egFront/OpenData/downloadXML.jsp>
- 教育部主管法規共用系統範例：<https://edu.law.moe.gov.tw/LawContent.aspx?id=FL036208>
- 臺北市法規查詢系統附件範例：<https://laws.gov.taipei/Law/LawSearch/LawFileList/FL029544>
- 全國法規資料庫資料檔：<https://data.gov.tw/dataset/18289>
