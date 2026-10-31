import pytest

from calc import BadExpression, evaluate, normalize


def test_evaluates_plain_and_typographic_arithmetic():
    assert evaluate("23.2 × 1.25") == 29.0
    assert evaluate("60 * 0.77") == pytest.approx(46.2)
    assert evaluate("3,000 × 1 ＋ 47,000 × 0.35") == pytest.approx(19450.0)
    assert evaluate("（10 ＋ 15 × 0.4）") == 16.0
    assert evaluate("60") == 60.0 and evaluate("-5 + 8") == 3.0
    assert normalize("２３.２ × １.２５") == "23.2*1.25"
    assert evaluate("23.2 × 1.25 = 29") == 29.0  # the appended result is ignored
    assert evaluate("47,000 × 35% + 3,000") == pytest.approx(19450.0)
    assert evaluate("60 × 0.77 ≈ 46.2") == pytest.approx(46.2)


def test_rejects_everything_that_is_not_arithmetic():
    for bad in (
        "",
        "   ",
        "2 ** 10",
        "__import__('os')",
        "abs(3)",
        "x * 2",
        "1/0",
        "9" * 201,
        "60 A",
        "表二五～二",
    ):
        with pytest.raises(BadExpression):
            evaluate(bad)
