"""build_conditions on the built variants: the parse breaks where it should, the witness labels
do not, and the clean variant reproduces 01's chunk counts."""

from pathlib import Path

import pytest

from run_text import VARIANTS, build_conditions, load_line_maps, load_questions, summarize

HERE = Path(__file__).parents[1]
ONE = HERE.parent / "01-chunking"


@pytest.fixture(scope="module")
def built():
    if not (HERE / "corpus/text/concat.txt").exists():
        pytest.skip("corpus not built")
    tables = {p.stem: p.read_text("utf-8") for p in (ONE / "corpus/tables").glob("*.txt")}
    maps = load_line_maps()
    out = {}
    for v, (fname, lm) in VARIANTS.items():
        out[v] = build_conditions((HERE / "corpus" / fname).read_text("utf-8"), maps[lm], tables)
    return out


def test_clean_matches_01(built):
    conds, stats = built["clean"]
    assert stats["articles_parsed"] == 1014 and stats["article"]["n"] == 1050
    assert len(conds["article"]) == 1050 + 109 and stats["article"]["parse_labels_agree"] == 1.0
    assert stats["article"]["unlabelled"] == 0


def test_plumber_two_column_breaks_the_parse_but_not_the_labels(built):
    _, stats = built["two-column/plumber"]
    assert stats["articles_parsed"] < 0.5 * 1014
    assert stats["fixed-1000"]["unlabelled"] < 0.1 * stats["fixed-1000"]["n"]


def test_pypdf_two_column_and_header_footer_keep_the_parse(built):
    for v in ("two-column/pypdf", "header-footer"):
        _, stats = built[v]
        assert stats["articles_parsed"] == 1014, v


def test_concat_distractor_chunks_carry_no_law_labels(built):
    conds, stats = built["concat"]
    assert stats["articles_parsed"] > 1014  # the distractor's 第 N 條 lines are parsed too
    art = [c for c in conds["article"] if "table" not in c.meta]
    assert stats["article"]["unlabelled"] >= 286  # the distractor has 286 articles
    # every question's gold article is still a label somewhere
    labels = {lab for c in art for lab in c.labels}
    assert all(set(q["gold"]) <= labels for q in load_questions())


def test_summarize_pairs_each_variant_against_clean():
    raw = []
    for v in ("clean", "concat"):
        for ret in ("dense", "random"):
            for i in range(3):
                raw.append(
                    {
                        "variant": v,
                        "chunking": "article",
                        "retriever": ret,
                        "qid": f"q{i}",
                        "hit@1000": 1.0 if v == "clean" else 0.0,
                    }
                )
    summary, diffs = summarize(raw, ["hit@1000"])
    assert len(summary) == 4
    assert [d["variant"] for d in diffs] == ["concat"] and diffs[0]["diff"] == -1.0
