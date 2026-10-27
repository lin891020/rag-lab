from fetch_corpus import parse_attachments

HTML = """
<a href="LawGetFile.ashx?FileId=0000395915&amp;lan=C" target="_blank">表八 低壓用電設備最小工作空間.PDF</a>
<a href="LawGetFile.ashx?FileId=0000395917&amp;lan=C" target="_blank">表二五～一 導線絕緣物之最高容許溫度.PDF</a>
<a href="LawGetFile.ashx?FileId=0000396041&amp;lan=C" target="_blank">圖五九七～四 無揮發氣控制示意圖.PDF</a>
"""


def test_parse_attachments_keeps_tables_drops_figures():
    got = parse_attachments(HTML)
    assert got == [
        ("0000395915", "表八", "表八 低壓用電設備最小工作空間"),
        ("0000395917", "表二五～一", "表二五～一 導線絕緣物之最高容許溫度"),
    ]
