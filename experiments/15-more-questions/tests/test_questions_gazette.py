"""The new gazette bank: 09's checks, plus the gold number inside the quote, plus the counts
the README reports (quote coverage per route, repeats, article-header quotes)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from questions_check import number_in_quote, occurrences, route_texts, starts_with_article_header
from run_retrieval import contains

HERE = Path(__file__).parents[1]
NINE = HERE.parent / "09-messy-real"
TEN = HERE.parent / "10-fix-ablation"
BANK = HERE / "questions-gazette.jsonl"
pytestmark = pytest.mark.skipif(not BANK.exists(), reason="bank not written yet")
ROUTES = {
    "base": NINE / "corpus/text",
    "ocr": NINE / "corpus/text",
    "rows": NINE / "corpus/text",
    "ocrrows": TEN / "corpus/text",
}


def _bank() -> list[dict]:
    return [json.loads(ln) for ln in BANK.read_text("utf-8").splitlines() if ln.strip()]


def test_ids_fields_and_types():
    qs = _bank()
    ids = [q["id"] for q in qs]
    assert ids == [f"q{n}" for n in range(161, 161 + len(qs))], "ids q161… in order"
    docs = {
        d["sysid"] for d in json.loads((NINE / "corpus/manifest.json").read_text("utf-8"))["docs"]
    }
    for q in qs:
        assert q["doc"] in docs, q["id"]
        assert q["type"] in ("value", "articles", "which"), q["id"]
        assert len(q["quote"].replace(" ", "")) >= 6, q["id"]
        assert "「" in q["question"] or q["type"] == "which", q["id"]  # 09's rule
        e = q["expect"]
        if q["type"] == "value":
            assert isinstance(e["value"], (int, float)) and not isinstance(e["value"], bool), q[
                "id"
            ]
            assert isinstance(e["unit"], str), q["id"]
        elif q["type"] == "articles":
            assert e["articles"] and all(isinstance(a, str) for a in e["articles"]), q["id"]
        else:
            assert isinstance(e["law"], str) and e["law"] and e["law"] not in q["question"], q["id"]


def test_every_document_has_questions_and_the_bank_is_160():
    qs = _bank()
    per = Counter(q["doc"] for q in qs)
    assert len(per) == 30 and min(per.values()) >= 5
    assert len(qs) == 160


def test_quotes_are_verbatim_and_hold_the_gold_number():
    missing, no_number = [], []
    for q in _bank():
        texts = route_texts(ROUTES, q["doc"])
        if not any(contains(t, q["quote"]) for r, t in texts.items() if r in ("base", "ocr")):
            missing.append(q["id"])
        if q["type"] == "value" and not number_in_quote(
            q["quote"], q["expect"]["value"], q["expect"]["unit"]
        ):
            no_number.append(q["id"])
    assert not missing, missing
    assert not no_number, no_number


def test_counts_for_the_readme(capsys):
    """Not assertions: the numbers README §2 reports (printed with -s)."""
    qs = _bank()
    cover = Counter()
    repeats = 0
    headers = 0
    for q in qs:
        texts = route_texts(ROUTES, q["doc"])
        for r, t in texts.items():
            cover[r] += contains(t, q["quote"])
        repeats += occurrences(texts.get("ocr", texts.get("base", "")), q["quote"]) > 1
        headers += starts_with_article_header(q["quote"])
    print(
        {
            "coverage": dict(cover),
            "repeated_quotes": repeats,
            "article_header_quotes": headers,
            "types": dict(Counter(q["type"] for q in qs)),
        }
    )
