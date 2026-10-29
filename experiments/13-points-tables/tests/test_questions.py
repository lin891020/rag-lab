"""The question bank: well-formed, every quote found in its document's extracted text (base,
else ocr, whitespace-insensitive), every document covered."""

import json
from collections import Counter
from pathlib import Path

import pytest

from run_retrieval import contains

HERE = Path(__file__).parents[1]
QUESTIONS = [
    json.loads(ln)
    for ln in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    if ln.strip()
]
MANIFEST = json.loads((HERE / "corpus/manifest.json").read_text(encoding="utf-8"))
DOC_IDS = {d["sysid"] for d in MANIFEST["docs"]}


def test_ids_unique_and_fields_well_formed():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    for q in QUESTIONS:
        assert q["doc"] in DOC_IDS, q["id"]
        assert q["type"] in ("value", "articles", "which"), q["id"]
        assert q["kind"] in ("points", "tables"), q["id"]
        assert len(q["quote"].replace(" ", "")) >= 6, q["id"]
        assert "「" in q["question"] or q["type"] == "which", q["id"]
        e = q["expect"]
        if q["type"] == "value":
            assert isinstance(e["value"], (int, float)) and isinstance(e["unit"], str), q["id"]
        elif q["type"] == "articles":
            assert e["articles"] and all(isinstance(a, str) for a in e["articles"]), q["id"]
        else:
            assert isinstance(e["law"], str) and e["law"], q["id"]


def test_every_document_has_questions_and_the_bank_is_big_enough():
    per_doc = Counter(q["doc"] for q in QUESTIONS)
    assert set(per_doc) == DOC_IDS
    assert min(per_doc.values()) >= 3
    assert len(QUESTIONS) >= 150


@pytest.mark.skipif(not (HERE / "corpus/text").exists(), reason="corpus not extracted")
def test_every_quote_is_in_its_document_text():
    missing = []
    for q in QUESTIONS:
        found = False
        for route in ("base", "ocr"):
            p = HERE / "corpus/text" / f"{q['doc']}.{route}.txt"
            if p.exists() and contains(p.read_text(encoding="utf-8"), q["quote"]):
                found = True
                break
        if not found:
            missing.append((q["id"], q["doc"], q["quote"]))
    assert not missing, missing
