"""21's tool answer: for each lookup, the whole column of that table (every row, the asked row
marked), once per (table, column); a lookup whose column is unknown gets its whole row; a miss of
both is said in words. The model still picks bands and does the arithmetic: no rule is called."""

from lookup import TABLES, render_segment, resolve
from rules import row_key


def test_a_band_lookup_shows_every_band_of_that_column():
    text, labels = render_segment([resolve("表五二", "大賣場(倉儲) 100000", "需量因數")])
    assert "表五二｜大賣場(倉儲) 12,500以下｜" in text
    assert "表五二｜大賣場(倉儲) 超過12,500｜" in text and "← 你查的" in text
    keys = {f"表五二|{row_key('表五二', r)}" for r in TABLES["表五二"]["rows"]}
    assert labels == keys


def test_only_the_asked_row_is_marked():
    text, _ = render_segment([resolve("表九三～二", "700 A", "絞線")])
    marked = [ln for ln in text.splitlines() if ln.endswith("← 你查的")]
    assert len(marked) == 1 and "表九三～二｜800以下｜" in marked[0]


def test_the_same_column_twice_is_shown_once():
    one, _ = render_segment([resolve("表五三", "10以下", "需量因數")])
    two, _ = render_segment(
        [resolve("表五三", "10以下", "需量因數"), resolve("表五三", "超過10", "需量因數")]
    )
    rows = lambda t: [ln for ln in t.splitlines() if ln.startswith("表五三｜")]  # noqa: E731
    assert len(rows(one)) == len(rows(two)) == 2
    assert sum(ln.endswith("← 你查的") for ln in rows(two)) == 2


def test_an_unknown_row_still_gets_the_column():
    text, labels = render_segment([resolve("表五三", "沒有這列", "需量因數")])
    assert "表五三｜10以下｜" in text and "← 你查的" not in text
    assert "表五三|超過10" in labels


def test_an_unknown_column_gets_the_whole_row():
    text, labels = render_segment([resolve("表二五～五", "絞線 8", "沒有這欄")])
    row = [ln for ln in text.splitlines() if ln.startswith("表二五～五｜絞線 8｜")]
    assert len(row) == len(TABLES["表二五～五"]["names"])
    assert labels == {"表二五～五|絞線 8"}


def test_misses_are_said_in_words():
    text, labels = render_segment(
        [resolve("表九九", "x", "y"), resolve("表五三", "沒有這列", "沒有這欄")]
    )
    assert (
        "查無此表：表九九" in text and "查無此列：沒有這列" in text and "查無此欄：沒有這欄" in text
    )
    assert labels == set()


def test_notes_and_title_come_once_per_table():
    text, _ = render_segment(
        [
            resolve("表五二", "住宅 3,000", "需量因數"),
            resolve("表五二", "醫院 50,000以下", "需量因數"),
        ]
    )
    assert text.count("表五二 註") <= 1
    assert sum(ln.startswith("表五二 ") and "需量因數" in ln for ln in text.splitlines()[:1]) == 1
