"""The detection evaluation on the fixtures: real continuations accepted, a foreign first page
rejected, and the bookkeeping counts add up."""

from pathlib import Path

from detect import evaluate, first_failing_check
from tables import load_pages

FIX = Path(__file__).parents[3] / "tests/fixtures"


def test_evaluate_on_fixtures():
    grids = {
        "0000395929": load_pages(FIX / "table56.pdf"),
        "0000395996": load_pages(FIX / "table587-1.pdf"),
        "0000395998": load_pages(FIX / "table587-3.pdf"),
        "0000395999": load_pages(FIX / "table8.pdf"),
    }
    out = evaluate(grids)
    assert (out["positives"], out["negatives"]) == (4, 3)
    assert out["tp"] == 4 and out["fp"] == 0
    assert out["n_pairs"] == 7


def test_first_failing_check_names_the_title():
    a = load_pages(FIX / "table587-3.pdf")[1]  # a page whose table does reach the bottom
    b = load_pages(FIX / "table8.pdf")[0]
    import dataclasses

    b = dataclasses.replace(b, page_no=a.page_no + 1)
    assert first_failing_check(a, b) == "cur_has_title_or_not_at_top"
