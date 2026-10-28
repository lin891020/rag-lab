"""Chunk labels that do not depend on parse_articles surviving the mess: a chunk is labelled
with every article that has at least one typeset line (>= 8 chars) inside the chunk."""

from pathlib import Path

from chunkers import Chunk, article_chunks, fixed_chunks, parse_articles
from labels import label_chunks, line_map_from_text
from rag_lab.pdf import extract_text

LAW = Path(__file__).parents[2] / "01-chunking/corpus/law.pdf"


def test_line_map_assigns_every_line_to_the_running_article():
    text = "   第 一 章 總則\n第 1 條\n甲甲甲甲甲甲甲甲甲甲\n第 2 條\n乙乙乙乙乙乙乙乙乙乙\n丙丙丙丙丙丙丙丙丙丙\n"
    lm = line_map_from_text(text)
    assert lm == [
        ("   第 一 章 總則", ""),
        ("第 1 條", "1"),
        ("甲甲甲甲甲甲甲甲甲甲", "1"),
        ("第 2 條", "2"),
        ("乙乙乙乙乙乙乙乙乙乙", "2"),
        ("丙丙丙丙丙丙丙丙丙丙", "2"),
    ]


def test_label_chunks_uses_line_witnesses_and_ignores_short_lines():
    lm = [
        ("第 1 條", "1"),
        ("甲甲甲甲甲甲甲甲甲甲", "1"),
        ("乙乙乙乙乙乙乙乙乙乙", "2"),
        ("短", "3"),
    ]
    chunks = [
        Chunk("頁首\n甲甲甲甲甲甲甲甲甲甲\n乙乙乙乙乙乙乙乙乙乙 短", []),
        Chunk("沒有任何見證行", []),
    ]
    out = label_chunks(chunks, lm)
    assert out[0].labels == ["1", "2"] and out[1].labels == []
    assert out[0].text == chunks[0].text  # text untouched


def test_on_the_real_pdf_line_labels_agree_with_01s_parse():
    text = extract_text(LAW)
    lm = line_map_from_text(text)
    articles = parse_articles(text)
    assert len(articles) == 1014 and len(lm) > 5000
    art = label_chunks([Chunk(c.text, []) for c in article_chunks(articles)], lm)
    ref = article_chunks(articles)
    # every article chunk gets its own number back (header line is a witness), nothing else
    assert sum(a.labels == r.labels for a, r in zip(art, ref, strict=True)) == len(ref)
    fixed = fixed_chunks(
        text,
        size=1000,
        overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
        spans=[(a.start, a.end, str(a.num)) for a in articles],
    )
    relabelled = label_chunks([Chunk(c.text, []) for c in fixed], lm)
    agree = sum(set(a.labels) == set(r.labels) for a, r in zip(relabelled, fixed, strict=True))
    assert (
        agree / len(fixed) > 0.85
    )  # a span that merely touches a chunk is a parse label, not a witness
