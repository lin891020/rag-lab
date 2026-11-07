"""The question set -> questions.jsonl, from corpus/rows.json (the official comparison table).

    uv run python experiments/29-amendment-diff/make_questions.py            # write questions.jsonl
    uv run python experiments/29-amendment-diff/make_questions.py --t3-candidates  # rows for T3

Every question quotes the new article (114-05-09) in full and asks about the 110-03-17 version:
  T1  which old article (or 新增): from rows mapping one new to one whole old article, plus
      additions (new only). 60 = 50 + 10, stratified by chapter.
  T2  was the article's text amended: 沒改 when textnorm.canon of the two cells is equal (numerals
      read as values, references and list markers masked, punctuation dropped), 有改 when it
      differs AND the note names a change (修正／增訂／刪除／整併／酌修…). Any wording change
      counts (the question says so); renumbering, references, numeral style and punctuation do not. Rows whose text differs while the note only says
      條次變更 are left out and counted (the plan's "both signals must agree"). As many 有改 as
      there are 沒改 (20 + 20 here: this amendment rewrote nearly every article).
  T3  a value that changed: candidates are rows whose value multisets differ (textnorm.numbers_of)
      after a human picks the quantity; the picks live in corpus/t3-manual.json (checked by an
      independent agent); at most 30 — 12 here, the rest of the candidates only renumber,
      restyle (六○○ -> 六百), convert units (40 公分 -> 400 毫米) or drop a value.
The sample is random.Random(0) over rows sorted by new article number. Stratified: each chapter
gets round(share × n), at least one where it has rows. questions-stats.json keeps the counts.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from rag_lab.log import setup_logging

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import laws  # noqa: E402
import textnorm as tn  # noqa: E402
from compare_table import ROWS, sort_key  # noqa: E402

logger = logging.getLogger(__name__)

NEW_TXT = HERE / "corpus/new-1140509.txt"
T3_MANUAL = HERE / "corpus/t3-manual.json"
OUT = HERE / "questions.jsonl"
STATS = HERE / "corpus/questions-stats.json"
SEED = 0
T1_N, T1_ADDED = 60, 10
T2_HALF_MAX = 30
T3_MAX = 30
CHANGE_WORDS = ("修正", "增訂", "新增", "刪除", "整併", "酌修", "酌作", "調整", "移列至", "合併")
NEW_DATE, OLD_DATE = "114 年 5 月 9 日", "110 年 3 月 17 日"

Q_T1 = (
    "下面是《用戶用電設備裝置規則》新版（{new_date}修正）第 {n} 條的全文。"
    "這一條在舊版（{old_date}修正）是第幾條？舊版沒有對應的條文（本條是新增的）就回答沒有對應。"
    "\n\n新版第 {n} 條：\n{text}"
)
Q_T2 = (
    "下面是《用戶用電設備裝置規則》新版（{new_date}修正）第 {n} 條的全文。"
    "跟它在舊版（{old_date}修正）對應的那一條比，條文的文字有沒有被修改？"
    "用詞、句子、規定事項的任何改動都算修改；只有條號不同、引據的條號與表號跟著重排、"
    "數字寫法不同（六○○與六百）、標點與分項分款的編號不同，不算修改。"
    "\n\n新版第 {n} 條：\n{text}"
)
Q_T3 = (
    "下面是《用戶用電設備裝置規則》新版（{new_date}修正）第 {n} 條的全文。"
    "它規定的「{quantity}」，舊版（{old_date}修正）是多少、新版是多少？單位：{unit}；"
    "原文用別的單位時請換算。兩個數值都只填數字。"
    "\n\n新版第 {n} 條：\n{text}"
)


def load_rows() -> list[dict]:
    return [r for r in json.loads(ROWS.read_text("utf-8")) if r["kind"] == "article"]


def new_articles() -> dict[str, laws.LawArticle]:
    return {a.num: a for a in laws.parse_law(NEW_TXT.read_text("utf-8"))}


def note_says_changed(note: str) -> bool:
    n = note.replace("\n", "").replace(" ", "")
    return any(w in n for w in CHANGE_WORDS)


def classify(rows: list[dict]) -> dict[str, list[dict]]:
    """one_to_one / added / t2_same / t2_changed / t2_ambiguous (text differs, note silent)."""
    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if len(r["new_nums"]) != 1:
            continue
        if (
            not r["old_nums"] and not r["old_text"].strip()
        ):  # an added article: the old cell is empty
            out["added"].append(r)
            continue
        if not r["old_nums"]:
            continue  # the old cell has text but no header we could read: not a question
        if len(r["old_nums"]) != 1 or r.get("old_part"):
            continue
        out["one_to_one"].append(r)
        if tn.canon(r["new_text"]) == tn.canon(r["old_text"]):
            out["t2_same"].append(r)
        elif note_says_changed(r["note"]):
            out["t2_changed"].append(r)
        else:
            out["t2_ambiguous"].append(r)
    return out


def stratified(
    rows: list[dict], n: int, chapters: dict[str, str], rng: random.Random
) -> list[dict]:
    """n rows, each chapter round(share × n) and at least one where it has rows; sorted by new
    article number before drawing so the draw depends on the seed alone."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: sort_key(r["new_nums"][0])):
        by[chapters[r["new_nums"][0]]].append(r)
    total = sum(len(v) for v in by.values())
    quota = {ch: max(1, round(len(v) / total * n)) for ch, v in by.items()}
    while sum(quota.values()) > n:  # rounding overshoot: trim the biggest
        quota[max(quota, key=lambda c: (quota[c], c))] -= 1
    while sum(quota.values()) < n:
        quota[max(by, key=lambda c: (len(by[c]) - quota[c], c))] += 1
    picked = []
    for ch in sorted(by):
        picked += rng.sample(by[ch], min(quota[ch], len(by[ch])))
    return sorted(picked, key=lambda r: sort_key(r["new_nums"][0]))


def t3_candidates(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if tn.canon(r["new_text"]) == tn.canon(r["old_text"]):
            continue
        a, b = Counter(tn.numbers_of(r["new_text"])), Counter(tn.numbers_of(r["old_text"]))
        if a != b:
            out.append(
                {
                    "new_num": r["new_nums"][0],
                    "old_num": r["old_nums"][0],
                    "only_old": sorted((b - a).elements()),
                    "only_new": sorted((a - b).elements()),
                    "new_text": r["new_text"],
                    "old_text": r["old_text"],
                    "note": r["note"],
                }
            )
    return out


_PARA = re.compile(r"^\d{1,2}\s{2,}", re.M)  # the PDF's margin paragraph numbers 「1　依本規則…」


def shown_text(body: str) -> str:
    """The new article as the model sees it: the PDF's paragraph numbers dropped (the old
    version's page and the comparison table have none, so T2 would read them as a change)."""
    return _PARA.sub("", body)


def _q(
    kind: str, i: int, r: dict, art: laws.LawArticle, question: str, expect: dict, old: str | None
) -> dict:
    return {
        "id": f"{kind.lower()}-{i:03d}",
        "kind": kind,
        "chapter": art.chapter,
        "new_num": r["new_nums"][0],
        "old_num": old,
        "new_text": shown_text(art.body),
        "question": question,
        "expect": expect,
        "gold_labels": [f"old:{old}"] if old else [],
        "row_pages": r["pages"],
    }


def build() -> list[dict]:
    rng = random.Random(SEED)
    rows = load_rows()
    arts = new_articles()
    chapters = {n: a.chapter for n, a in arts.items()}
    cls = classify(rows)
    qs: list[dict] = []

    t1 = stratified(cls["one_to_one"], T1_N - T1_ADDED, chapters, rng) + stratified(
        cls["added"], T1_ADDED, chapters, rng
    )
    for i, r in enumerate(sorted(t1, key=lambda r: sort_key(r["new_nums"][0])), 1):
        n = r["new_nums"][0]
        old = r["old_nums"][0] if r["old_nums"] else None
        qs.append(
            _q(
                "T1",
                i,
                r,
                arts[n],
                Q_T1.format(
                    new_date=NEW_DATE, old_date=OLD_DATE, n=n, text=shown_text(arts[n].body)
                ),
                {"old_num": old},
                old,
            )
        )

    half = min(len(cls["t2_same"]), len(cls["t2_changed"]), T2_HALF_MAX)
    same = (
        cls["t2_same"]
        if len(cls["t2_same"]) <= half
        else stratified(cls["t2_same"], half, chapters, rng)
    )
    changed = stratified(cls["t2_changed"], half, chapters, rng)
    t2 = sorted(same + changed, key=lambda r: sort_key(r["new_nums"][0]))
    same_ids = {id(r) for r in same}
    for i, r in enumerate(t2, 1):
        n = r["new_nums"][0]
        qs.append(
            _q(
                "T2",
                i,
                r,
                arts[n],
                Q_T2.format(
                    new_date=NEW_DATE, old_date=OLD_DATE, n=n, text=shown_text(arts[n].body)
                ),
                {"changed": id(r) not in same_ids},
                r["old_nums"][0],
            )
        )

    if T3_MANUAL.exists():
        byrow = {(r["new_nums"][0], r["old_nums"][0]): r for r in cls["one_to_one"]}
        picks = json.loads(T3_MANUAL.read_text("utf-8"))
        for i, p in enumerate(sorted(picks, key=lambda p: sort_key(p["new_num"]))[:T3_MAX], 1):
            r = byrow[(p["new_num"], p["old_num"])]
            n = p["new_num"]
            # one row may give two questions (two values changed): ids stay unique by index
            qs.append(
                _q(
                    "T3",
                    i,
                    r,
                    arts[n],
                    Q_T3.format(
                        new_date=NEW_DATE,
                        old_date=OLD_DATE,
                        n=n,
                        text=shown_text(arts[n].body),
                        quantity=p["quantity"],
                        unit=p.get("unit", ""),
                    ),
                    {
                        "quantity": p["quantity"],
                        "old_value": p["old_value"],
                        "new_value": p["new_value"],
                        "unit": p.get("unit", ""),
                    },
                    p["old_num"],
                )
            )
    return qs


def stats(rows: list[dict], qs: list[dict]) -> dict:
    cls = classify(rows)
    return {
        "article_rows": len(rows),
        "one_to_one": len(cls["one_to_one"]),
        "added": len(cls["added"]),
        "t2_same": len(cls["t2_same"]),
        "t2_changed": len(cls["t2_changed"]),
        "t2_ambiguous_text_differs_note_silent": len(cls["t2_ambiguous"]),
        "t3_candidates": len(t3_candidates(cls["one_to_one"])),
        "questions": dict(Counter(q["kind"] for q in qs)),
        "t1_added": sum(q["kind"] == "T1" and q["expect"]["old_num"] is None for q in qs),
        "seed": SEED,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3-candidates", action="store_true", help="write corpus/t3-candidates.json")
    args = ap.parse_args(argv)
    setup_logging("rag_lab")
    rows = load_rows()
    if args.t3_candidates:
        cands = t3_candidates(classify(rows)["one_to_one"])
        (HERE / "corpus/t3-candidates.json").write_text(
            json.dumps(cands, ensure_ascii=False, indent=1) + "\n", "utf-8"
        )
        logger.info("t3 candidates", extra={"ctx": {"n": len(cands)}})
        return 0
    qs = build()
    OUT.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), "utf-8")
    s = stats(rows, qs)
    STATS.write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", "utf-8")
    logger.info("questions", extra={"ctx": s})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
