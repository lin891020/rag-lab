# 天生就亂的繁中公開文件：找到什麼、沒找到什麼（2026-10-29）

> Mike 的第 ③ 項：08 用人工弄亂，之後要換天生就亂的真資料。這份是找來源的紀錄。
> 每個候選都實際抓了樣本、跑 `experiments/08-messy/doc_health.py` 體檢；樣本在當次 session 的 scratchpad，不進 git。

## 一、結論先講

**有標註正解的繁中「亂文件」評測集不存在。** OmniDocBench v1.5 的 1,651 頁裡繁中只有 13 頁（5 研究報告、4 古籍、2 考卷、2 書）；JVZY（Scientific Data 2024）是隨機字元合成的退化圖，不是真文件；Hugging Face 上的繁中文件集不是古籍手稿就是 30 份沒標註的 PDF。所以跟 01–08 一樣：**來源拿公開的真文件，題目自己出。**

最合用的來源是**行政院公報的法規修正案**：跟 01–08 同領域（法規）、天生就亂（三欄「修正條文／現行條文／說明」對照表跨頁、每格是整段條文、每頁有頁首）、能整批抓（國家圖書館開放資料 XML 給每筆 PDF 連結）。

## 二、候選總表

| 來源 | 怎麼拿 | 樣本體檢（doc_health） | 亂在哪 | 跟 Mike 領域的距離 | 判定 |
|---|---|---|---|---|---|
| **行政院公報／六都公報**（政府公報資訊網開放資料） | `https://gaz.ncl.edu.tw/opendata/gov.xml`（14 MB、11,403 筆、10,438 筆有 PDF、每半年更新；這版只涵蓋 2024-07～12）；每筆 `downFile.jsp?sysid=…&flag=P` | 行政院公報 10 頁：18 頁有表、頁首「行政院公報 第#卷 第#期 # 農業環保篇」；產生器 iText／Word 混 | 修正條文對照表：三欄、跨頁、格內是整段條文；公告附表；地方公報的公示送達名單 | 近：法規，而且半年就有 1,210 筆「修正條文／規則／辦法／標準」有檔，含「輸配電業各項費率計算公式」「職業安全衛生設施規則」「消防安全檢查注意事項」 | **第一候選** |
| **立法院公報**（ly.govapi.tw v2 → `公報完整PDF網址`） | `https://ly.govapi.tw/v2/gazettes` → agendas → ppg.ly.gov.tw 的完整 PDF；另有 doc 原檔與 txt／html 處理版（= 現成的文字正解） | 第 115 卷 77 期 451 頁：47 頁有表、39 頁對齊文字（出席名單、表決名單）、頁首每頁 | 名單式對齊文字、質詢逐字稿、預算表；同一份裡文字／表／名單混 | 遠：議事 | 備用；優點是有官方 txt 可當文字層正解 |
| **考選部考畢試題**（data.gov.tw 170565 → CSV 67,088 筆、101 年起） | `wwwc.moex.gov.tw/main/Exam/wHandExamQandA_CSV.ashx`；每題 PDF 與標準答案 PDF 連結 | 101 年測驗題 4 頁：3 頁對齊文字（四個選項排成欄）、頁首「代號：#｜頁次：#－#」；Acrobat Distiller | 選項欄位對齊、表頭區塊、數學式；舊年度掃描與否沒驗完（抓到的 101 年是文字層） | 遠：考題不是文件問答 | 不用；但「題目 + 標準答案」是現成標註，若要測 OCR 字元正確率可用 |
| **監察院調查報告**（cybsbox.cy.gov.tw） | 單檔下載，沒找到批次 | 133 頁：57 頁有表、Word 2010、直式橫書 | 長報告、附表、引文；相對乾淨 | 中：有「台電配電外線帶料發包」這種案子 | 備用 |
| **公共工程施工綱要規範**（工程會） | 官方站 pcces.pcc.gov.tw 這台機器解析不到；第三方（treca.org.tw）有單章 | 第 01521 章 24 頁：0 頁有表（lines 策略）、頁尾「# #-# V#.# #/#/#」 | Word 匯出、條列式、附會議紀錄 | 近：施工規範 | 來源不穩，先不用 |
| **環評書件查詢系統**（eiadoc.moenv.gov.tw） | 前端是 SPA（`/public/documentDetail/{HCode}`），檔案走 JS 呼叫的 API，curl 抓不到；開放資料 API 要 api_key | 沒抓到樣本 | 傳聞最亂：數百頁、掃描與文字混、地圖、工程表 | 中：環工 | 要用得先用瀏覽器看 API；這次沒做 |
| **政府公報舊期**（1930 教育部公報） | detail 頁有 `twinfo.ncl.edu.tw/.../merge_pdf.hpg` 連結，但不在開放資料 XML 裡 | 3 頁：0 頁有文字層（純掃描、直書） | 純影像、直書、老字體 | 遠 | 不用 |
| 臺北市法規查詢系統附件（laws.taipei.gov.tw/lawatt） | curl 只拿到 1 KB HTML（要瀏覽器 cookie） | 沒抓到 | 附件常是掃描表 | 近 | 要用得過瀏覽器 |

## 三、建議的 09

- **語料**：從 gov.xml 挑經濟部／勞動部／內政部的「修正條文」公告 20–30 份（對照表式），一份 10–50 頁。
- **題目**：「第 N 條修正後規定什麼」「這次改了哪幾條」「說明欄給的理由」——正解直接從對照表的欄位抄，可以自己核。
- **會撞到的**：對照表三欄跨頁（05 的問題）、格內整段文字（03／04 沒遇過）、頁首每頁（08 證明無害，可驗一次）、產生器混雜（iText vs Word）。先跑 `doc_health.py` 再決定要不要加 OCR 軌。
- **先問 Mike**：領域要不要限制在電業／職安，還是全部門都收。

## 四、搜了什麼、沒找到

- 「繁體中文 文件 OCR 資料集 benchmark」「zh-TW document layout dataset」「Traditional Chinese table recognition dataset」：只有場景文字（esun TC-STR）、古籍（AncientDoc、羅家倫手稿 921 頁）、合成（JVZY、ZihCiLin 410 萬張）。
- 台電購料規範／設備規範書：沒有公開下載頁，只有年報與永續報告。
- 智慧局專利公報全文影像：收費（FTP／DVD）；開放資料只有書目 XML。
- 公開資訊觀測站財報：要走表單下載，沒驗。

## 五、來源連結

- 政府公報資訊網開放資料：https://data.gov.tw/dataset/8178 ・ https://gaz.ncl.edu.tw/opendata/gov.xml
- 立法院 API：https://ly.govapi.tw/v2/gazettes ・ https://data.ly.gov.tw/
- 考選部試題：https://data.gov.tw/dataset/170565
- 環評書件：https://eiadoc.moenv.gov.tw/ ・ 操作手冊 https://eiadoc.moenv.gov.tw/files/operation_public.pdf ・ API https://data.moenv.gov.tw/swagger/
- OmniDocBench：https://huggingface.co/datasets/opendatalab/OmniDocBench（`OmniDocBench.json` 42 MB，`page_attribute.language` 數出 13 頁繁中）
- JVZY：https://arxiv.org/abs/2411.16387 ・ https://doi.org/10.1038/s41597-024-04146-7
- 其他：https://github.com/esun-ai/traditional-chinese-text-recogn-dataset ・ https://huggingface.co/datasets/ZihCiLin/traditional-chinese-ocr-synthetic ・ https://huggingface.co/datasets/KAI-KratosAI/Chinese_Documents_Dataset_PDF
