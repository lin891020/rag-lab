from attribute import cause_of, numbers_in, required_inputs, value_present

Q3 = {
    "id": "c3",
    "given": {"hp": 15, "volts": 380, "conductors": 3},
    "inputs": {"flc": 23.2, "factor": 1.25, "required": 29.0, "ampacity": 33.0},
    "expect": {"kind": "ladder", "value": 8.0, "unit": "mm²", "ladder": [3.5, 5.5, 8.0, 14.0]},
}
Q2 = {
    "id": "c2",
    "given": {"insulation": 90, "size": 22.0, "conductors": 3, "ambient": 45},
    "inputs": {"ampacity": 60.0, "temp_factor": 0.77},
    "expect": {"kind": "number", "value": 46.2, "unit": "A", "ladder": None},
}
QA = {
    "id": "ca",
    "given": {},
    "inputs": {},
    "expect": {"kind": "abstain", "value": None, "unit": "A", "ladder": None},
}


def _r(
    cond="L1-struct",
    *,
    right=False,
    answered=True,
    grade="error",
    rows_hit=2,
    rows_total=2,
    expression="",
    quote="",
    value=0.0,
):
    return {
        "condition": cond,
        "right": right,
        "answered": answered,
        "grade": grade,
        "rows_hit": rows_hit,
        "rows_total": rows_total,
        "expression": expression,
        "quote": quote,
        "value": value,
    }


def test_numbers_and_intermediate_values():
    assert numbers_in("23.2 × 1.25 = 29，再查 3,000 以下") == [23.2, 1.25, 29.0, 3000.0]
    assert required_inputs(Q3) == [
        23.2
    ]  # ladder: the chosen row's ampacity is not required and required_inputs(Q2) == [60.0, 0.77]
    # a table value that coincides with a number the question gives is not required (conservative)
    assert required_inputs({**Q2, "given": {**Q2["given"], "insulation": 60}}) == [0.77]
    assert (
        value_present(0.77, [77.0])
        and value_present(46.2, [46000.0])
        and not value_present(60.0, [61.0])
    )


def test_cause_order():
    assert cause_of(_r(right=True), Q2) is None
    assert cause_of(_r(answered=True), QA) == "hard_answer"
    assert cause_of(_r(answered=False, grade="abstained"), Q2) == "wrong_abstain"
    assert cause_of(_r(rows_hit=1), Q2) == "retrieval_miss"
    # 17: a missing gold row is reported before the abstention it caused
    assert cause_of(_r(rows_hit=1, answered=False, grade="abstained"), Q2) == "retrieval_miss"
    assert cause_of(_r(expression="55 × 0.77", quote="絞線 22"), Q2) == "value_copy"  # 60 missing
    assert cause_of(_r(expression="", quote="60 0.77"), Q2) == "no_expression"
    assert cause_of(_r(expression="60 + 0.77", quote="60 0.77"), Q2) == "formula"
    assert cause_of(_r(expression="60 × 0.77", quote="", value=40.0), Q2) == "arithmetic"
    assert cause_of(_r(expression="60 × 0.77", grade="match"), Q2) == "unit_only"
    assert (
        cause_of(_r(expression="23.2 × 1.25", quote="33", grade="critical", value=5.5), Q3)
        == "selection"
    )
    assert (
        cause_of(_r(expression="23.2 × 1.5", quote="33 29", grade="error", value=14.0), Q3)
        == "formula"
    )
