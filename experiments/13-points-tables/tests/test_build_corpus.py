"""The filter (要點-style vs table-only notices), the strata, and 09's documents going first."""

from build_corpus import MINIMA, PER_TYPE, doc_type, parse_records, pick, wanted

XML = """<?xml version="1.0" encoding="UTF-8" ?>
<OAIXML>
<Record><gazette>
  <Sysid><![CDATA[P1]]></Sysid>
  <Caption><![CDATA[修正「宜蘭縣政府工程績效獎金實施要點」部分條文，自113年11月8日生效]]></Caption>
  <GazName><![CDATA[宜蘭縣政府公報]]></GazName>
  <PubDate><![CDATA[20241108]]></PubDate>
  <Pages><![CDATA[頁10-12]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=P1&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
<Record><gazette>
  <Sysid><![CDATA[T1]]></Sysid>
  <Caption><![CDATA[修正「桃園市犬貓收容及處理收費標準」第三條附表]]></Caption>
  <GazName><![CDATA[桃園市政府公報]]></GazName>
  <PubDate><![CDATA[20240901]]></PubDate>
  <Pages><![CDATA[頁1-3]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=T1&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
<Record><gazette>
  <Sysid><![CDATA[A1]]></Sysid>
  <Caption><![CDATA[勞動部令：修正「職業安全衛生設施規則」部分條文]]></Caption>
  <GazName><![CDATA[行政院公報]]></GazName>
  <PubDate><![CDATA[20240901]]></PubDate>
  <Pages><![CDATA[頁1-8]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=A1&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
<Record><gazette>
  <Sysid><![CDATA[L1]]></Sysid>
  <Caption><![CDATA[訂定「新北市政府體育局績效評量基準」，並自113年7月22日生效]]></Caption>
  <GazName><![CDATA[新北市政府公報]]></GazName>
  <PubDate><![CDATA[20240722]]></PubDate>
  <Pages><![CDATA[頁1-40]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=L1&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
</OAIXML>
"""


def test_doc_type_and_wanted():
    recs = {r["sysid"]: r for r in parse_records(XML)}
    assert doc_type(recs["P1"]) == "points" and wanted(recs["P1"])
    assert doc_type(recs["T1"]) == "tables" and wanted(recs["T1"])
    assert doc_type(recs["A1"]) is None and not wanted(recs["A1"])  # 條文: 09's kind, not ours
    assert doc_type(recs["L1"]) == "points" and not wanted(recs["L1"])  # 40 pages


def test_pick_fills_both_strata_and_takes_the_first_list_first():
    pool = [
        {
            "sysid": f"P{i}",
            "caption": "修正「X要點」",
            "gazette": "新北市政府公報",
            "file": "u",
            "pages": 3,
        }
        for i in range(40)
    ] + [
        {
            "sysid": f"T{i}",
            "caption": "修正「Y收費標準」第三條附表",
            "gazette": "行政院公報",
            "file": "u",
            "pages": 3,
        }
        for i in range(20)
    ]

    def features(r):
        return {
            "type": "points" if r["sysid"].startswith("P") else "tables",
            "local": r["gazette"] != "行政院公報",
            "producer": "itext" if int(r["sysid"][1:]) % 2 else "word",
            "scanned": r["sysid"] == "P7",
            "garbage": False,
        }

    chosen, _ = pick(pool, features=features, seed=0, first=[pool[3], pool[5]])
    assert sum(c["type"] == "points" for c in chosen) == PER_TYPE["points"]
    assert sum(c["type"] == "tables" for c in chosen) == PER_TYPE["tables"]
    assert {"P3", "P5"} <= {c["sysid"] for c in chosen}
    assert sum(c["local"] for c in chosen) >= MINIMA["local"]
    assert any(c["scanned"] for c in chosen)
