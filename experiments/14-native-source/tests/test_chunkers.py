"""Points (一、二、) at the start of an unindented line are chunk boundaries; 款 (（一）) and 目
(  1.) stay inside their point; text before the first point is kept as "pre" chunks."""

from chunkers import parse_points, points_chunks

TEXT = """宜蘭縣政府公報 113 年 11 月 第 392期
修正「宜蘭縣政府工程績效獎金實施要點」第四點，並自113年11月8日生效。
一、宜蘭縣政府為獎勵工程人員，特訂定本要點。
二、本要點適用對象如下：
（一）本府各單位實際從事工程業務之人員。
（二）約用人員。
  1.約用人員。
  2.非第二點第一款所列聘僱人員。
三、本會置委員七人至十一人，其中一人為主任委員，由縣長或其指定人員擔任，其餘委員
  由本府就下列人員遴聘（派）兼之：
（一）本府社會處代表一人。
四、提撥上限：
（一）每人每年度最高提撥新臺幣五萬五千元整。
十一、本會所需經費，由本府社會處編列預算支應。
"""


def test_parse_points_finds_top_level_points_only():
    pts = parse_points(TEXT)
    assert [p.label for p in pts] == ["1", "2", "3", "4", "11"]
    assert "（一）本府各單位" in pts[1].body and "2.非第二點" in pts[1].body  # 款 and 目 kept
    assert pts[2].body.startswith("三、本會置委員七人至十一人")  # marker kept: quotes start with it
    assert "由本府就下列人員遴聘" in pts[2].body  # continuation line joined
    assert pts[0].start < pts[1].start < pts[4].end <= len(TEXT)


def test_points_chunks_keep_the_front_matter_and_label_points():
    chunks = points_chunks(TEXT, max_len=1000)
    assert (
        chunks[0].labels == ["pre"]
        and "修正「宜蘭縣政府工程績效獎金實施要點」第四點" in chunks[0].text
    )
    pts = [c for c in chunks if c.labels != ["pre"]]
    assert [c.labels for c in pts] == [["1"], ["2"], ["3"], ["4"], ["11"]]
    assert pts[3].text.startswith("第 4 點\n四、提撥上限")


def test_long_points_split_at_款_and_repeat_the_header():
    long = (
        "一、總則：\n" + "\n".join(f"（{c}）" + "甲" * 300 for c in "一二三四五") + "\n二、短的。\n"
    )
    chunks = points_chunks(long, max_len=700)
    first = [c for c in chunks if c.labels == ["1"]]
    assert len(first) == 3 and all(c.text.startswith("第 1 點\n") for c in first)
    assert all(len(c.text) <= 720 for c in first)


def test_no_points_means_no_chunks():
    assert parse_points("只有一段沒有編號的文字。\n第二段。") == []
    assert points_chunks("只有一段沒有編號的文字。\n第二段。", max_len=1000) == []
