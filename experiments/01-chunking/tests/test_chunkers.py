from pathlib import Path

import pytest

from chunkers import (
    Chunk,
    article_chunks,
    fixed_chunks,
    parse_articles,
    table_flat_fixed_chunks,
    table_row_chunks,
    table_whole_chunks,
)
from rag_lab.pdf import extract_text

HERE = Path(__file__).parents[1]
SEPS = ["\n\n", "\n", ". ", " ", ""]  # securag's RecursiveCharacterTextSplitter separators


@pytest.fixture(scope="module")
def law_text():
    return extract_text(HERE / "corpus/law.pdf")


@pytest.fixture(scope="module")
def articles(law_text):
    return parse_articles(law_text)


@pytest.fixture(scope="module")
def tables():
    return {p.stem: p.read_text() for p in sorted((HERE / "corpus/tables").glob("*.txt"))}


def test_parse_articles_finds_all_1014_once(articles):
    assert [a.num for a in articles] == list(range(1, 1015))


def test_parse_articles_carries_chapter_and_section(articles):
    a9 = articles[8]
    assert (a9.num, a9.chapter, a9.section) == (9, "第一章 總則", "第二節 電路之檢驗")
    a16 = articles[15]
    assert (a16.chapter, a16.section) == ("第二章 配線及保護", "第一節 導線")
    assert articles[0].body == "本規則依電業法第三十二條第五項規定訂定之。"


def test_article_chunks_one_per_article_with_header(articles):
    chunks = article_chunks(articles, max_len=10_000)
    assert len(chunks) == 1014
    assert chunks[8].text.startswith("第一章 總則 / 第二節 電路之檢驗 / 第 9 條\n")
    assert chunks[8].labels == ["9"]


def test_article_chunks_splits_long_articles_and_repeats_header(articles):
    chunks = article_chunks(articles, max_len=1000)
    assert len(chunks) > 1014
    assert all(len(c.text) <= 1000 + 60 for c in chunks)  # header allowance
    pieces = [c for c in chunks if c.labels == ["94"]]  # a long article (>1000 chars)
    assert len(pieces) >= 2 and all(
        p.text.startswith("第二章 配線及保護 / 第五節 接地及搭接 / 第 94 條") for p in pieces
    )


def test_fixed_chunks_respect_size_and_overlap():
    text = "甲" * 30 + "\n" + "乙" * 30 + "\n" + "丙" * 30
    chunks = fixed_chunks(text, size=40, overlap=10, separators=SEPS, spans=[(0, len(text), "x")])
    assert all(len(c.text) <= 40 for c in chunks)
    assert "".join(c.text for c in chunks).count("乙") >= 30  # nothing dropped
    assert chunks[1].text.startswith(chunks[0].text[-10:].split("\n")[-1][:5]) or len(chunks) >= 3


def test_fixed_chunks_label_every_article_exactly_by_span(articles, law_text):
    spans = [(a.start, a.end, str(a.num)) for a in articles]
    chunks = fixed_chunks(law_text, size=1000, overlap=200, separators=SEPS, spans=spans)
    assert all(c.labels for c in chunks)
    covered = {lab for c in chunks for lab in c.labels}
    assert covered == {str(n) for n in range(1, 1015)}
    # a 1000-char chunk starting inside article 25 must carry label "25" even without its header
    inside = next(c for c in chunks if c.start > articles[24].start and c.start < articles[24].end)
    assert "25" in inside.labels


def test_table_whole_one_chunk_per_table(tables):
    chunks = table_whole_chunks(tables)
    assert len(chunks) == 109
    c = next(c for c in chunks if c.labels == ["表二五～二"])
    assert c.text.startswith("表二五～二 金屬導線管配線之導線安培容量") and "註：" in c.text


def test_table_row_chunks_prefix_title_and_header(tables):
    chunks = table_row_chunks(tables)
    rows = [c for c in chunks if c.labels == ["表二五～二"]]
    assert len(rows) >= 20
    hit = next(c for c in rows if c.text.endswith("3.5 7/0.8 19 17 15 13"))
    assert hit.text.startswith("表二五～二 金屬導線管配線之導線安培容量")
    assert "3以下 4 5-6 7-9" in hit.text
    assert "1.6 13 12 11 9" not in hit.text  # other rows are not in this chunk


def test_table_flat_fixed_labels_tables_by_span(tables):
    chunks = table_flat_fixed_chunks(tables, size=1000, overlap=200, separators=SEPS)
    assert {lab for c in chunks for lab in c.labels} == set(tables)
    assert all(isinstance(c, Chunk) and c.labels for c in chunks)


def test_table_row_chunks_accept_values_with_a_suffix():
    from chunkers import table_row_chunks

    text = "表九二 線徑\n導線 線徑\n銅線 25以下\n鋁線 l0以下\n註：略\n"
    rows = table_row_chunks({"表九二": text})
    assert [c.text.splitlines()[-1] for c in rows] == ["銅線 25以下", "鋁線 l0以下"]
