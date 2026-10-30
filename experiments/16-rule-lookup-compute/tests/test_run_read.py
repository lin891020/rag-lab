"""Evidence without any model: the shared corpus text and its row-level spans, the two cuts,
oracle evidence holding the gold rows, gold judged by labels, the L2 tool override."""

import json

from chunkers import Chunk
from run_read import (
    Evidence,
    apply_tool,
    budget_evidence,
    corpus_text,
    fixed_corpus_chunks,
    gold_labels,
    load_corpus,
    load_questions,
    rule_excerpt,
    struct_chunks,
)
from rules import TABLES

Q = load_questions()


def test_corpus_text_spans_label_articles_and_clean_rows():
    law, raw = load_corpus()
    text, spans = corpus_text(law, raw)
    labels = {lab for _, _, lab in spans}
    assert (
        "25" in labels
        and "214" in labels
        and "表二五～五|絞線 22" in labels
        and "表九三～二|100以下" in labels
    )
    s, e, _ = next(sp for sp in spans if sp[2] == "表二五～五|絞線 22")
    assert text[s:e].startswith("導線線徑／線別=絞線；導線線徑／標稱截面積(mm2)=22")
    assert len(raw) == 109 - len(TABLES)


def test_both_cuts_carry_row_level_labels():
    law, raw = load_corpus()
    struct = struct_chunks(law, raw)
    fixed = fixed_corpus_chunks(law, raw)
    want = "表二五八～三|15"
    assert any(want in c.labels for c in struct) and any(want in c.labels for c in fixed)
    assert any("表五二|註" in c.labels and "手術室" in c.text for c in struct)
    assert any("表五二|註" in c.labels for c in fixed)
    row_chunk = next(c for c in struct if want in c.labels)
    assert row_chunk.text.startswith("表二五八～三") and "380=23.2" in row_chunk.text
    assert all(len(c.text) <= 1000 for c in fixed)


def test_oracle_holds_every_gold_row_and_the_rule():
    ev = Evidence(Q, conditions=["L1-oracle"], embed=lambda t: [], build=False)
    q = next(x for x in Q if x["chain"] == "C3" and x["expect"]["kind"] == "ladder")
    text, label, hit = ev.for_question("L1-oracle", q)
    assert "【第 214 條】" in text and "一‧二五倍" in text
    for tid, key in q["gold"]["rows"]:
        assert TABLES[tid]["title"] in text, (tid, key)
    assert hit == {"rows_hit": 2, "rows_total": 2, "article_hit": True}


def test_budget_evidence_judges_gold_by_labels_and_reads_the_first_chunk():
    gold = {"表二五～五|絞線 22", "表二五～七|41 - 45"}
    ranked = [Chunk("x" * 1600, ["表二五～五|絞線 22"]), Chunk("y" * 100, ["表二五～七|41 - 45"])]
    text, labels = budget_evidence(ranked, gold)
    assert (
        "表二五～五|絞線 22" in labels and "表二五～七|41 - 45" not in labels
    )  # first cut, second did not fit
    assert text.startswith("【段 1】")


def test_apply_tool_overrides_number_values_only():
    q_num = {"expect": {"kind": "number"}}
    q_lad = {"expect": {"kind": "ladder"}}
    content = json.dumps(
        {"found": True, "quote": "x", "expression": "23.2 × 1.25", "value": 1, "unit": "A"}
    )
    out, ok = apply_tool(content, q_num)
    assert ok and json.loads(out)["value"] == 29.0
    out, ok = apply_tool(content, q_lad)
    assert ok and json.loads(out)["value"] == 1
    out, ok = apply_tool(
        json.dumps({"found": False, "quote": "", "expression": "", "value": 0, "unit": ""}), q_num
    )
    assert not ok and json.loads(out)["value"] == 0
    out, ok = apply_tool(
        json.dumps({"found": True, "quote": "x", "expression": "abs(2)", "value": 7, "unit": "A"}),
        q_num,
    )
    assert not ok and json.loads(out)["value"] == 7


def test_rule_excerpt_centres_on_the_table_mention():
    body = "甲" * 1000 + "表二五～七" + "乙" * 1000
    ex = rule_excerpt(body, ["表二五～七"], chars=300)
    assert "表二五～七" in ex and len(ex) == 300
    assert gold_labels({"gold": {"rows": [["表五三", "10以下"]], "articles": ["53"]}}) == (
        {"表五三|10以下"},
        {"53"},
    )
