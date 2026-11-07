"""make_questions: T1 from one-to-one rows plus additions, T2 from rows whose canon text is equal
(沒改) or differs with the note naming a change (有改), both halves the same size, T3 from the
hand-checked file; every question carries the new article's text, the gold labels and the
chapter; the sample is stratified by chapter and fixed by seed 0."""

import json
import re

import make_questions as mq
import run

Q = run.load_questions()


def test_kinds_and_counts():
    by = {k: [q for q in Q if q["kind"] == k] for k in ("T1", "T2", "T3")}
    assert (
        len(by["T1"]) == mq.T1_N
        and sum(q["expect"]["old_num"] is None for q in by["T1"]) == mq.T1_ADDED
    )
    changed = [q["expect"]["changed"] for q in by["T2"]]
    assert changed.count(True) == changed.count(False) == len(by["T2"]) // 2
    assert 0 < len(by["T3"]) <= mq.T3_MAX
    assert len({q["id"] for q in Q}) == len(Q)


def test_every_question_carries_the_new_text_gold_labels_and_chapter():
    for q in Q:
        assert not re.match(r"^第 \d+(-\d+)? 條", q["new_text"])  # the body, no header line
        assert q["chapter"].startswith("第") and "章" in q["chapter"]
        assert q["question"].count(f"新版第 {q['new_num']} 條") == 1
        if q["kind"] == "T1":
            assert q["gold_labels"] == (
                [] if q["expect"]["old_num"] is None else [f"old:{q['expect']['old_num']}"]
            )
        else:
            assert q["gold_labels"] == [f"old:{q['old_num']}"]


def test_t2_gold_is_text_based_with_the_note_agreeing_on_changes():
    rows = {tuple(r["new_nums"]): r for r in mq.load_rows() if len(r["new_nums"]) == 1}
    for q in Q:
        if q["kind"] != "T2":
            continue
        r = rows[(q["new_num"],)]
        same = mq.tn.canon(r["new_text"]) == mq.tn.canon(r["old_text"])
        assert q["expect"]["changed"] == (not same)
        if not same:
            assert mq.note_says_changed(r["note"])


def test_shown_text_drops_the_pdfs_paragraph_numbers():
    assert mq.shown_text("1   甲乙。\n2   丙丁。\n一、戊") == "甲乙。\n丙丁。\n一、戊"
    assert mq.shown_text("1.第一款\n10 條") == "1.第一款\n10 條"  # not a margin number
    assert not any(re.match(r"^\d{1,2}\s{2,}", q["new_text"]) for q in Q)


def test_added_t1_rows_have_an_empty_old_cell_and_the_note_says_new():
    rows = {tuple(r["new_nums"]): r for r in mq.load_rows() if len(r["new_nums"]) == 1}
    for q in Q:
        if q["kind"] == "T1" and q["expect"]["old_num"] is None:
            r = rows[(q["new_num"],)]
            assert r["old_text"].strip() == "" and "新增" in r["note"], (q["id"], q["new_num"])


def test_t1_only_one_to_one_rows_or_additions():
    rows = {tuple(r["new_nums"]): r for r in mq.load_rows() if len(r["new_nums"]) == 1}
    for q in Q:
        if q["kind"] != "T1":
            continue
        r = rows[(q["new_num"],)]
        assert len(r["old_nums"]) <= 1 and not r.get("old_part")
        assert (q["expect"]["old_num"] is None) == (r["old_nums"] == [])


def test_same_seed_same_questions(tmp_path):
    a = mq.build()
    b = mq.build()
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


def test_stratified_every_chapter_with_rows_gets_at_least_one(tmp_path):
    chapters = {q["chapter"] for q in Q if q["kind"] == "T1"}
    assert len(chapters) >= 5
