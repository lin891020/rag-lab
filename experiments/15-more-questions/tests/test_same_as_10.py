"""The evidence builder and the ruler are 10's: the merge of old and new banks is only honest
if both were fed and scored the same way. Byte-identical files, or identical top-level blocks
where 15 added a flag around them."""

import re
from pathlib import Path

HERE = Path(__file__).parents[1]
TEN = HERE.parent / "10-fix-ablation"


def _block(src: str, name: str) -> str:
    """Source of a top-level def/class up to the next top-level statement."""
    m = re.search(rf"^(?:def|class) {re.escape(name)}\b.*?(?=^\S)", src, re.S | re.M)
    assert m, name
    return m.group(0)


def test_shared_modules_are_byte_identical_to_10():
    for name in ("prompts.py", "chunkers.py", "retrievers.py", "run_retrieval.py", "extract.py"):
        assert (HERE / name).read_bytes() == (TEN / name).read_bytes(), name


def test_evidence_and_ruler_blocks_are_identical_to_10():
    mine, theirs = (
        (HERE / "run_read.py").read_text("utf-8"),
        (TEN / "run_read.py").read_text("utf-8"),
    )
    for name in (
        "budget_evidence",
        "oracle_window",
        "Evidence",
        "ask",
        "drift_sample",
        "done_keys",
    ):
        assert _block(mine, name) == _block(theirs, name), name
    for const in ("MODELS", "THINK", "OPTIONS", "BUDGET", "DRIFT_PER_CELL"):
        assert re.search(rf"^{const}\b.*$", mine, re.M).group(0) == re.search(
            rf"^{const}\b.*$", theirs, re.M
        ).group(0), const
    mine, theirs = (HERE / "score.py").read_text("utf-8"), (TEN / "score.py").read_text("utf-8")
    for name in (
        "judge",
        "judge_value",
        "judge_articles",
        "judge_which",
        "numeral_kind",
        "normalize_unit",
        "_unit_matches",
        "score_rows",
        "paired_diff_ci",
        "bootstrap_ci",
    ):
        assert _block(mine, name) == _block(theirs, name), name


def test_units_may_only_be_added():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ten_score", TEN / "score.py")
    ten = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ten)
    import score

    for unit, aliases in ten.UNIT_ALIASES.items():
        assert score.UNIT_ALIASES.get(unit) == aliases, unit
