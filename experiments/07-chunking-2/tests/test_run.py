"""build_conditions on the real law.pdf: sizes, labels, and the small-to-big owner map."""

from pathlib import Path

from pypdf import PdfReader
from rag_lab.pdf import extract_text
from run import build_conditions, load_questions

ONE = Path(__file__).parents[2] / "01-chunking"


def test_conditions_on_the_real_pdf():
    law = ONE / "corpus/law.pdf"
    text = extract_text(law)
    pages = [p.extract_text() or "" for p in PdfReader(str(law)).pages]
    tables = {p.stem: p.read_text("utf-8") for p in (ONE / "corpus/tables").glob("*.txt")}
    conds, stats = build_conditions(text, pages, tables, {})
    art, _, _ = conds["article"]
    key = next(f"19:{i}" for i, c in enumerate(art) if c.meta.get("article") == 19)
    conds, stats = build_conditions(text, pages, tables, {key: "馬達分路導線"})
    assert stats["article"]["n"] == 1050 + 109  # 01's counts
    assert stats["page"]["n"] <= 283 + 109
    nop, _, _ = conds["article-noprefix"]
    assert not nop[0].text.startswith("第一章") and nop[0].labels == art[0].labels
    ctx, _, _ = conds["article+context"]
    assert sum(c.text.startswith("馬達分路導線\n") for c in ctx) == 1
    small, parents, owner = conds["small-to-big"]
    assert owner is not None and len(owner) == len(small) and len(parents) == len(art)
    assert all(set(c.labels) == set(parents[o].labels) for c, o in zip(small, owner, strict=True))
    assert max(len(c.text) for c in small) <= 269
    # a page chunk carries the article running over from the previous page
    page_chunks, _, _ = conds["page"]
    assert any(len(c.labels) > 1 and c.text.find(f"第 {c.labels[0]} 條") == -1 for c in page_chunks)


def test_questions_are_01s_article_questions_plus_reserve():
    qs = load_questions()
    assert len(qs) == 40 and all(q["type"] == "article" for q in qs)


def test_load_questions_takes_extra_files_and_still_keeps_only_article_questions(tmp_path):
    extra = tmp_path / "more.jsonl"
    extra.write_text(
        '{"id": "a41", "type": "article", "gold": ["20"]}\n'
        '{"id": "t18", "type": "table", "gold": ["表一"]}\n',
        "utf-8",
    )
    qs = load_questions([ONE / "questions.jsonl", extra])
    assert qs[-1]["id"] == "a41" and all(q["type"] == "article" for q in qs)
