from chunkers import Chunk
from metrics import bootstrap_ci, hit_at_budget, hit_at_k, paired_diff_ci

RANKED = [Chunk("x" * 600, ["3"]), Chunk("y" * 600, ["7"]), Chunk("z" * 100, ["9", "10"])]


def test_hit_at_k_any_gold_label_in_top_k():
    assert hit_at_k(RANKED, ["7"], k=2) is True
    assert hit_at_k(RANKED, ["7"], k=1) is False
    assert hit_at_k(RANKED, ["10", "99"], k=3) is True


def test_hit_at_budget_counts_only_chunks_that_fit_whole():
    assert hit_at_budget(RANKED, ["3"], budget=600) is True
    assert hit_at_budget(RANKED, ["7"], budget=1000) is False  # second chunk would end at 1200
    assert hit_at_budget(RANKED, ["7"], budget=1200) is True
    # the first chunk is always read (cut to the budget), as 10's budget_evidence does
    assert hit_at_budget(RANKED, ["3"], budget=100) is True
    assert hit_at_budget(RANKED, ["7"], budget=100) is False


def test_bootstrap_ci_degenerate_and_ordered():
    assert bootstrap_ci([1.0] * 10, n=200, seed=0) == (1.0, 1.0, 1.0)
    mean, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], n=500, seed=0)
    assert lo <= mean <= hi and 0.3 < lo and hi < 1.0


def test_paired_diff_ci_sign_and_zero_when_identical():
    a = [1, 1, 1, 0, 1, 0, 1, 1]
    b = [0, 1, 0, 0, 1, 0, 0, 1]
    d, lo, hi = paired_diff_ci(a, b, n=500, seed=0)
    assert abs(d - 0.375) < 1e-9 and lo > 0
    assert paired_diff_ci(a, a, n=100, seed=0) == (0.0, 0.0, 0.0)
