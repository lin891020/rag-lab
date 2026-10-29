"""Record parsing, the filter, the strata, and the quota-filling pick (no network)."""

from build_corpus import (
    MINIMA,
    PER_TYPE,
    doc_type,
    is_local,
    parse_records,
    pick,
    producer_family,
    wanted,
)

XML = """<?xml version="1.0" encoding="UTF-8" ?>
<OAIXML>
<Record><gazette>
  <Sysid><![CDATA[E1]]></Sysid>
  <Caption><![CDATA[農業部公告：預告「漁船船員管理規則」部分條文修正草案]]></Caption>
  <GazName><![CDATA[行政院公報]]></GazName>
  <PubDate><![CDATA[20240806]]></PubDate>
  <Pages><![CDATA[頁37551-37560]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=E1&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
<Record><gazette>
  <Sysid><![CDATA[E2]]></Sysid>
  <Caption><![CDATA[新北市政府第697次市政會議紀錄]]></Caption>
  <GazName><![CDATA[新北市政府公報]]></GazName>
  <PubDate><![CDATA[20240717]]></PubDate>
  <Pages><![CDATA[頁1-4]]></Pages>
  <Edoc><EFile/><EFormat/></Edoc>
</gazette></Record>
<Record><gazette>
  <Sysid><![CDATA[E3]]></Sysid>
  <Caption><![CDATA[修正「基隆市環境保護局組織規程」部分條文]]></Caption>
  <GazName><![CDATA[基隆市政府公報]]></GazName>
  <PubDate><![CDATA[20240601]]></PubDate>
  <Pages><![CDATA[頁9-12]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=E3&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
<Record><gazette>
  <Sysid><![CDATA[E4]]></Sysid>
  <Caption><![CDATA[勞動部令：修正「職業安全衛生設施規則」部分條文]]></Caption>
  <GazName><![CDATA[行政院公報]]></GazName>
  <PubDate><![CDATA[20240901]]></PubDate>
  <Pages><![CDATA[頁1-80]]></Pages>
  <Edoc><EFile><![CDATA[https://gaz.ncl.edu.tw/downFile.jsp?sysid=E4&flag=P]]></EFile><EFormat><![CDATA[PDF]]></EFormat></Edoc>
</gazette></Record>
</OAIXML>
"""


def test_parse_records_reads_the_fields_we_use():
    recs = parse_records(XML)
    assert [r["sysid"] for r in recs] == ["E1", "E2", "E3", "E4"]
    assert recs[0]["pages"] == 10 and recs[1]["pages"] == 4
    assert recs[1]["file"] == "" and recs[0]["file"].endswith("sysid=E1&flag=P")


def test_wanted_keeps_amendments_with_a_file_and_3_to_30_pages():
    recs = parse_records(XML)
    assert [r["sysid"] for r in recs if wanted(r)] == ["E1", "E3"]  # E2 no file, E4 80 pages


def test_doc_type_and_locality():
    recs = {r["sysid"]: r for r in parse_records(XML)}
    assert doc_type(recs["E1"]) == "draft" and doc_type(recs["E3"]) == "final"
    assert not is_local(recs["E1"]) and is_local(recs["E3"])


def test_producer_family():
    assert producer_family("iText® 5.5.13 ©2000-2018 iText Group NV") == "itext"
    assert producer_family("iLovePDF") == "ilovepdf"
    assert producer_family("Microsoft® Word 2010") == "word"
    assert producer_family("適用於 Microsoft 365 的 Microsoft® Word") == "word"
    assert producer_family("Acrobat Distiller 9.0.0 (Windows)") == "distiller"
    assert producer_family("") == "other"


def _fake_pool(n: int) -> list[dict]:
    fams = ["itext", "ilovepdf", "word", "distiller", "other"]
    pool = []
    for i in range(n):
        pool.append(
            {
                "sysid": f"S{i}",
                "type": "draft" if i % 2 == 0 else "final",
                "local": i % 3 == 0,
                "producer": fams[i % 5],
                "scanned": i % 17 == 0,  # a page without a text layer
                "garbage": i % 41 == 0,  # a page whose text is (cid:N) tokens
            }
        )
    return pool


def test_pick_fills_every_quota_and_minimum():
    pool = _fake_pool(200)
    chosen, scanned = pick(pool, features=lambda r: r, seed=0)
    assert len(chosen) == 2 * PER_TYPE
    assert sum(c["type"] == "draft" for c in chosen) == PER_TYPE
    assert sum(c["local"] for c in chosen) >= MINIMA["local"]
    assert sum(c["scanned"] for c in chosen) >= MINIMA["scanned"]
    assert sum(c["garbage"] for c in chosen) >= MINIMA["garbage"]
    for fam in ("itext", "ilovepdf", "word", "distiller"):
        assert sum(c["producer"] == fam for c in chosen) >= MINIMA["producer"]
    assert scanned >= len(chosen)  # it had to look at at least what it kept


def test_pick_is_deterministic_for_a_seed():
    pool = _fake_pool(200)
    a, _ = pick(pool, features=lambda r: r, seed=3)
    b, _ = pick(pool, features=lambda r: r, seed=3)
    assert [c["sysid"] for c in a] == [c["sysid"] for c in b]


def test_pick_reports_unmet_minimum_instead_of_looping_forever():
    pool = [dict(r, garbage=False) for r in _fake_pool(60)]
    chosen, _ = pick(pool, features=lambda r: r, seed=0)
    assert len(chosen) == 2 * PER_TYPE and sum(c["garbage"] for c in chosen) == 0
