from prompts import CONDITIONS


def test_thirteen_conditions_add_one_and_drop_one():
    assert len(CONDITIONS) == 13  # 2 references + fixed + 4 add-one + full + 5 drop-one
    assert CONDITIONS["fixed"] == ("base", "fixed-1000", False)
    assert CONDITIONS["full"] == ("ocrrows", "article-cn+pre", True)
    assert CONDITIONS["-ocr"] == ("rows", "article-cn+pre", True)  # 09's "all"
    # every add-one differs from fixed in exactly one of (route, chunking, prefix)
    for c in ("+ocr", "+rows", "+prefix", "+cnpre"):
        assert sum(a != b for a, b in zip(CONDITIONS[c], CONDITIONS["fixed"], strict=True)) == 1
    for c in ("-ocr", "-rows", "-prefix", "-pre", "-cnpre"):
        assert sum(a != b for a, b in zip(CONDITIONS[c], CONDITIONS["full"], strict=True)) == 1
