from merge import cluster_diff_ci, outcome_vs_prediction, pair_rows, paired_diff_ci, verdict


def test_paired_diff_ci_is_zero_when_identical_and_signed_otherwise():
    a = [1, 1, 1, 0, 1, 0, 1, 1]
    b = [0, 1, 0, 0, 1, 0, 0, 1]
    d, lo, hi = paired_diff_ci(a, b, n=500)
    assert abs(d - 0.375) < 1e-9 and lo > 0
    assert paired_diff_ci(a, a, n=100) == (0.0, 0.0, 0.0)
    assert verdict(lo, hi) == "分得出" and verdict(-0.1, 0.1) == "分不出"


def test_cluster_bootstrap_is_wider_when_the_effect_sits_in_few_clusters():
    # 4 clusters: two all +1, two all 0 -> the per-question CI is narrow, the cluster CI wide
    by_cluster = {"A": [1.0] * 10, "B": [1.0] * 10, "C": [0.0] * 10, "D": [0.0] * 10}
    d, lo, hi = cluster_diff_ci(by_cluster, n=500)
    flat = [x for v in by_cluster.values() for x in v]
    _, lo_q, hi_q = paired_diff_ci(flat, [0.0] * len(flat), n=500)
    assert abs(d - 0.5) < 1e-9 and (hi - lo) > (hi_q - lo_q)


def test_pair_rows_splits_old_new_and_pooled():
    a = {f"q{i}": i % 2 == 0 for i in range(40)}
    b = {f"q{i}": i % 3 == 0 for i in range(40)}
    old = {f"q{i}" for i in range(20)}
    new = {f"q{i}" for i in range(20, 40)}
    clusters = {f"q{i}": f"d{i // 5}" for i in range(40)}
    r = pair_rows(
        exp="x",
        model="m",
        pair="a − b",
        a=a,
        b=b,
        old_ids=old,
        new_ids=new,
        cluster_of=clusters,
        n_planned=40,
    )
    assert r["n_old"] == 20 and r["n_new"] == 20 and r["n_all"] == 40 and r["clusters"] == 8
    assert r["v_all"] in ("分得出", "分不出") and "mde_all" in r
    assert outcome_vs_prediction({"v_all": "分得出", "half_width_all": 0.05}, "會") == "會：如預期"
    assert (
        outcome_vs_prediction({"v_all": "分不出", "half_width_all": 0.02}, "收斂") == "收斂：如預期"
    )
    assert (
        outcome_vs_prediction({"v_all": "分得出", "half_width_all": 0.02}, "不會") == "不會：意外"
    )
