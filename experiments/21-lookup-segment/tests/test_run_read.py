"""21 re-asks only step 2 of 19's fix cells: the same step-1 reply (what the model pointed at),
the tool's answer widened to whole columns. 19's fix rows are copied; seg pairs with fix and seg-2
with fix-2 (gpt-oss), so the pointing is identical and only what the tool shows changes."""

import json

import run_read
from run_read import COPIED, RUNS, SOURCE, ask_seg, copied_rows, load_questions, step1_rows

Q = load_questions()
IDS = {q["id"] for q in Q}


def _fake(replies, prompts):
    it = iter(replies)

    def chat(prompt, model, schema):
        prompts.append(prompt)
        return {
            "content": next(it),
            "thinking": "",
            "eval_count": 5,
            "prompt_eval_count": 50,
            "duration": 0.5,
            "done_reason": "stop",
        }

    return chat


def test_cells_pair_with_19s_fix_cells():
    assert RUNS == {"qwen3:8b": ["seg"], "gpt-oss:20b": ["seg", "seg-2"]}
    assert SOURCE == {"seg": "fix", "seg-2": "fix-2"}
    for m, cells in RUNS.items():
        assert {SOURCE[c] for c in cells} <= set(COPIED[m])


def test_copied_rows_are_19s_fix_cells():
    rows = copied_rows(IDS)
    assert {r["reused_from"] for r in rows} == {"19"}
    assert len(rows) == 96 * 3 and {r["condition"] for r in rows} == {"fix", "fix-2"}


def test_every_seg_question_has_a_step1_reply():
    src = step1_rows(IDS)
    for m, cells in RUNS.items():
        for c in cells:
            assert all((m, SOURCE[c], qid) in src for qid in IDS), (m, c)


def test_ask_seg_reuses_step1_and_shows_whole_columns(monkeypatch):
    q = next(q for q in Q if q["id"] == "c4-20")
    src = step1_rows({q["id"]})[("gpt-oss:20b", "fix", q["id"])]
    prompts: list[str] = []
    reply = '{"found": true, "quote": "", "expression": "1", "value": 1, "unit": "W"}'
    monkeypatch.setattr(run_read, "_chat", _fake([reply], prompts))
    row = ask_seg("gpt-oss:20b", "seg", q, src)
    assert len(prompts) == 1 and row["error"] == "" and row["content"] == reply
    assert row["step1_content"] == src["step1_content"] and row["step1_from"] == "fix"
    assert row["lookups"] == src["lookups"]
    if any(x["table"] == "表五二" for x in src["lookups"]):
        assert "大賣場(倉儲) 12,500以下" in prompts[0] and "大賣場(倉儲) 超過12,500" in prompts[0]
    assert row["evidence_text"] in prompts[0] and row["rows_total"] >= 1


def test_a_failed_call_is_an_error_row(monkeypatch):
    def boom(prompt, model, schema):
        raise ValueError("bad")

    monkeypatch.setattr(run_read, "_chat", boom)
    src = step1_rows({Q[0]["id"]})[("qwen3:8b", "fix", Q[0]["id"])]
    row = ask_seg("qwen3:8b", "seg", Q[0], src)
    assert row["error"] and row["content"] == ""


def test_copied_fix_rows_count_needed_rows_only():
    rows = [r for r in copied_rows({"c4-16"})]
    assert rows and all(r["rows_total"] == 1 for r in rows)


def test_same_answer_compares_found_and_value():
    a = json.dumps({"found": True, "value": 2, "quote": "a"})
    b = json.dumps({"found": True, "value": 2, "quote": "b"})
    assert run_read.same_answer(a, b) and not run_read.same_answer(a, a.replace("2", "3"))


def test_a_gold_row_in_the_wrong_column_does_not_count_as_found(monkeypatch):
    """A whole column labels every row; the hit fields count a gold row only when a gold value
    is shown on it, so 'rows_all_hit' means the same as for fix (review 2026-11-03)."""
    q = next(q for q in Q if q["id"] == "c1-12")  # 單線 2.0, 8 conductors -> 16
    src = {
        "step1_content": "",
        "lookups": [{"asked": ["表二五～三", "單線 2.0", "4"]}],
    }
    reply = '{"found": true, "quote": "", "expression": "21", "value": 21, "unit": "A"}'
    monkeypatch.setattr(run_read, "_chat", _fake([reply], []))
    row = ask_seg("qwen3:8b", "seg", q, src)
    assert row["rows_hit"] == 0 and row["rows_total"] == 1
    src["lookups"].append({"asked": ["表二五～三", "單線 2.0", "8"]})
    monkeypatch.setattr(run_read, "_chat", _fake([reply], []))
    assert ask_seg("qwen3:8b", "seg", q, src)["rows_hit"] == 1
