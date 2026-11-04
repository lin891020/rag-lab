"""24's corpus is rebuilt from 23's saved glm-ocr replies, so no model is asked: glm2 must equal
23's glm2-s100 byte for byte, and glm2-flat may differ from it only in the 8 tables whose <thead>
has more than one row."""

from pathlib import Path

import pytest
from build_corpus import RAW23, build, headers, pages
from ocr import glm_to_markdown, header_rows, norm

HERE = Path(__file__).parents[1]
S100 = HERE.parent / "23-ocr-size/corpus/glm2-s100"
MULTI = {
    "表九三～二",
    "表二五～七",
    "表二五～二",
    "表二五～五",
    "表二五～四",
    "表五六",
    "表八八～二",
    "表八六～三",
}


@pytest.fixture(scope="module")
def built():
    return build(RAW23)


def _cells(line):
    return [norm(c) for c in line.strip().strip("|").split("|")]


def test_pages_grouped_per_table():
    p = pages(RAW23)
    assert len(p) == 17
    assert [x.stem for x in p["表五六"]] == ["表五六-p0", "表五六-p1"]


def test_glm2_is_23s_s100_byte_for_byte(built):
    assert set(built["glm2"]) == {f.stem for f in S100.glob("*.txt")}
    for tid, text in built["glm2"].items():
        assert text == (S100 / f"{tid}.txt").read_text("utf-8"), tid


def test_flat_differs_only_in_the_multi_row_headers(built):
    changed = {t for t in built["glm2"] if built["glm2-flat"][t] != built["glm2"][t]}
    assert changed == MULTI


def test_flat_drops_header_rows_and_keeps_every_other_line(built):
    for tid in MULTI:
        before = built["glm2"][tid].splitlines()
        after = built["glm2-flat"][tid].splitlines()
        new = [ln for ln in after if ln not in before]
        assert len(new) == 1, tid  # one flattened header (表五六's page 2 has none)
        assert [ln for ln in after if ln in before] == [
            ln for ln in before if ln in after
        ], tid  # the rest in the same order


def test_25_2_flattened_header_is_the_text_layers(built):
    flat = built["glm2-flat"]["表二五～二"].splitlines()[0]
    clean = next(
        ln
        for ln in (HERE / "corpus/clean/表二五～二.txt").read_text("utf-8").splitlines()
        if ln.startswith("|")
    )
    assert _cells(flat) == _cells(clean)


def test_headers_report_lists_the_multi_row_tables(built):
    rows = headers(RAW23, built, HERE / "corpus/clean")
    assert {r["table"] for r in rows} == MULTI
    # measured before the run: the 表二五 family matches the text layer; 表九三～二／五六／八八～二
    # differ by an OCR character (装、注、额) and 表八六～三's text-layer header is itself broken
    assert {r["table"] for r in rows if r["same_as_clean"]} == {
        "表二五～二",
        "表二五～四",
        "表二五～五",
        "表二五～七",
    }


def test_flatten_changes_the_first_page_of_the_8_tables_only():
    changed = set()
    for ps in pages(RAW23).values():
        for p in ps:
            h = p.read_text("utf-8")
            if glm_to_markdown(h, flatten=True) != glm_to_markdown(h):
                changed.add(p.stem)
    assert changed == {f"{t}-p0" for t in MULTI}


def test_per_page_flatten_replaces_exactly_the_thead_rows():
    for t in MULTI:
        h = (RAW23 / f"{t}-p0.html").read_text("utf-8")
        before, after = (
            glm_to_markdown(h).splitlines(),
            glm_to_markdown(h, flatten=True).splitlines(),
        )
        n = header_rows(h)

        def rows(lines):
            return [ln for ln in lines if ln.startswith("|") and not ln.startswith("|---")]

        assert rows(after)[1:] == rows(before)[n:], t  # the thead rows go, every other row stays
        assert rows(after)[0] not in before and after[1].startswith("|---"), t


def test_clean_is_23s_byte_for_byte():
    ours = sorted((HERE / "corpus/clean").glob("*.txt"))
    theirs = HERE.parent / "23-ocr-size/corpus/clean"
    assert len(ours) == 17
    for f in ours:
        assert f.read_bytes() == (theirs / f.name).read_bytes(), f.name
