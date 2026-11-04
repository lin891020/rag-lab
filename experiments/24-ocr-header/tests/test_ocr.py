"""06's HTML renderer as 23 had it, plus flatten: the rows of <thead> joined into one header row
the way 04 named the text layer's columns."""

from pathlib import Path

from ocr import glm_to_markdown, header_rows, html_to_markdown, norm

RAW23 = Path(__file__).parents[2] / "23-ocr-size/corpus/glm2-s100-raw"

HTML = (
    "<table><thead><tr><th rowspan='2'>a</th><th colspan='2'>b</th></tr>"
    "<tr><th>c</th><th>d</th></tr></thead>"
    "<tbody><tr><td>1</td><td>2</td><td>3</td></tr></tbody></table>"
)


def test_unflattened_output_is_23s():
    md = html_to_markdown(HTML)
    assert md.splitlines() == ["| a | b | b |", "|---|---|---|", "| a | c | d |", "| 1 | 2 | 3 |"]


def test_flatten_joins_thead_rows_skipping_a_repeat():
    md = html_to_markdown(HTML, flatten=True)
    assert md.splitlines() == ["| a | b／c | b／d |", "|---|---|---|", "| 1 | 2 | 3 |"]


def test_one_header_row_or_no_thead_is_unchanged():
    one = (
        "<table><thead><tr><th>a</th><th>b</th></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    )
    bare = "<table><tr><th>a</th></tr><tr><th>b</th></tr><tr><td>1</td></tr></table>"
    for h in (one, bare):
        assert html_to_markdown(h, flatten=True) == html_to_markdown(h)


def test_a_thead_without_a_spanning_cell_is_not_flattened():
    """表五六 page 2 is a continuation: glm-ocr put its first two data rows (26 - 30, 31 - 40) in
    <thead>. A header hierarchy has a parent cell spanning columns; rows without one stay."""
    h = (RAW23 / "表五六-p1.html").read_text("utf-8")
    assert header_rows(h) == 2
    assert html_to_markdown(h, flatten=True) == html_to_markdown(h)
    td = (
        "<table><thead><tr><td colspan='2'>a</td></tr><tr><td>b</td><td>c</td></tr></thead>"
        "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    )
    assert html_to_markdown(td, flatten=True).splitlines()[0] == "| a／b | a／c |"


def test_header_rows_of_23s_pages():
    multi = {
        "表二五～二-p0": 3,
        "表二五～四-p0": 3,
        "表二五～五-p0": 3,
        "表二五～七-p0": 2,
        "表九三～二-p0": 2,
        "表五六-p0": 2,
        "表五六-p1": 2,
        "表八八～二-p0": 2,
        "表八六～三-p0": 2,
    }
    pages = sorted(RAW23.glob("*.html"))
    assert len(pages) == 18
    for p in pages:
        h = p.read_text("utf-8")
        assert header_rows(h) == multi.get(p.stem, 1 if "<table" in h else 0), p.stem


def test_glm_to_markdown_passes_flatten_on_and_keeps_plain_text():
    plain = "表四○ 分路之導線\n絞線 3.5"
    assert glm_to_markdown(plain, flatten=True) == glm_to_markdown(plain) == plain
    assert glm_to_markdown(HTML, flatten=True) == html_to_markdown(HTML, flatten=True)


def test_norm_folds_width_and_spaces():
    assert norm("３８ - ５０") == norm("38-50") == "38-50"
    assert norm("（mm）") == "(mm)"
