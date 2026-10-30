from power import half_width, mde, n_needed, pair_stats, predict


def test_n_needed_matches_the_textbook_shape():
    # Connor 1987: 20% discordant, detect .05 -> about 620 pairs; detect .10 -> about 150
    assert 600 < n_needed(0.125, 0.075, 0.05) < 650
    assert 140 < n_needed(0.15, 0.05, 0.10) < 160
    assert n_needed(0.1, 0.1, 0.0) is None and n_needed(0.0, 0.0, 0.05) is None


def test_mde_inverts_n_needed_and_shrinks_with_n():
    assert abs(n_needed(0.21, 0.0, mde(0.21, 320)) - 320) < 1
    assert 0.06 < mde(0.21, 320) < 0.08  # the inspector's figure for +cnpre − fixed
    assert mde(0.21, 1000) < mde(0.21, 320) < mde(0.21, 160)
    assert mde(0.0, 100) is None


def test_pair_stats_and_prediction():
    a = {"q1": True, "q2": True, "q3": False, "q4": False}
    b = {"q1": True, "q2": False, "q3": True, "q4": False, "q5": True}  # q5 only on one side
    assert pair_stats(a, b) == (4, 0.25, 0.25)
    assert predict(0.0, 0.02, 320) == "收斂"  # a tight null
    assert predict(0.05, 0.21, 320) == "不會"
    assert predict(0.10, 0.21, 320) == "會"
    assert half_width(0.21, 0.0, 320) < half_width(0.21, 0.0, 160)
