"""The search tool indexes the old version alone: one label per old article (1,079, 13-1
included), every label prefixed old:, the new version's text nowhere in it; results are packed to
3,000 characters, the first unit cut when it alone does not fit, units that do not fit skipped."""

import laws
import search
from chunkers import Chunk


def test_index_labels_are_exactly_the_old_articles():
    units = search.build_units()
    labels = {lab for u in units for lab in u.labels}
    want = {f"old:{a.num}" for a in laws.parse_law(search.OLD_TXT.read_text("utf-8"))}
    assert labels == want and len(want) == 1079 and "old:13-1" in labels
    assert all(u.labels[0].startswith("old:") for u in units)


def test_the_new_version_is_not_in_the_index():
    units = search.build_units()
    new = laws.parse_law((search.HERE / "corpus/new-1140509.txt").read_text("utf-8"))
    a25 = next(a for a in new if a.num == "25")  # 低壓導線之安培容量…表二五～一 (new tables)
    assert not any("表二五～一" in u.text for u in units)
    assert not any(a25.body[:30] in u.text for u in units)


def test_units_repeat_the_header_and_stay_under_the_max():
    units = search.build_units()
    assert all(len(u.text) <= 1000 + 60 for u in units)  # header + a piece
    assert all(u.text.startswith("第") for u in units)


def test_budget_evidence_packs_and_cuts():
    big = Chunk("甲\n" * 2000, ["old:1"])
    small = Chunk("乙" * 100, ["old:2"])
    text, labels = search.budget_evidence([big, small], budget=500, pack=True)
    assert labels == {"old:1"} and len(text) <= 520  # the first unit alone is cut to the budget
    text, labels = search.budget_evidence(
        [small, big, Chunk("丙" * 50, ["old:3"])], budget=200, pack=True
    )
    assert labels == {"old:2", "old:3"}  # big skipped, the next one still taken


def test_index_sha_follows_the_built_index(monkeypatch):
    a = search.index_sha()
    search.index_sha.cache_clear()
    units = search.build_units()
    monkeypatch.setattr(search, "build_units", lambda: units[:-1])
    assert search.index_sha() != a
    search.index_sha.cache_clear()
