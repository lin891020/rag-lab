"""Which images each route reads, the cache that never re-asks for the same image, the image
token count kept per page, and the check that every route is complete before answering."""

import pytest
from read_sizes import PNG06, ROUTES, SHRUNK, page_dir, reply_format


def test_routes_are_the_planned_ones():
    assert list(ROUTES) == ["glm2-s100", "glm2-s83", "glm2-s67", "glm2-s50", "glm2-s33"]


def test_s100_is_06s_png_and_the_others_shrunk():
    assert page_dir("glm2-s100") == PNG06
    assert page_dir("glm2-s50") == SHRUNK / "s50"


def test_reply_format():
    assert reply_format("<table><tr><td>1</td></tr></table>") == "html"
    assert reply_format("表五二 需量因數 100") == "plain"


def _img(tmp_path, shade=0):
    from PIL import Image

    p = tmp_path / "表五二-p0.png"
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(p)
    return p


def test_cached_reply_keeps_image_tokens_and_is_reused_only_for_the_same_image(
    tmp_path, monkeypatch
):
    import read_sizes

    calls = []

    def fake(img, prompt):
        calls.append(img)
        return {
            "content": "<table><tr><td>1</td></tr></table>",
            "done_reason": "stop",
            "eval_count": 9,
            "prompt_eval_count": 2788,
        }

    monkeypatch.setattr(read_sizes, "glm_ocr", fake)
    img, raw = _img(tmp_path), tmp_path / "raw"
    md, info = read_sizes.read_page(img, raw)
    assert md == "| 1 |\n|---|" and info["format"] == "html"
    assert info["prompt_eval_count"] == 2788 and info["size"] == [8, 8]
    _, again = read_sizes.read_page(img, raw)
    assert len(calls) == 1 and again["cached"]
    _img(tmp_path, 200)  # another image under the same name
    with pytest.raises(SystemExit):
        read_sizes.read_page(img, raw)


def test_check_corpus_rejects_a_partial_or_missing_route():
    from read_sizes import check_corpus

    tids = ["表一", "表二"]
    pages = {"表一": 1, "表二": 2}
    stats = {"glm2-s50": {"表一": {"pages": 1}, "表二": {"pages": 2}}}
    check_corpus(["glm2-s50"], stats, tids, lambda r, t: pages[t])
    with pytest.raises(SystemExit):
        check_corpus(
            ["glm2-s50"], {"glm2-s50": {"表一": {"pages": 1}}}, tids, lambda r, t: pages[t]
        )
    with pytest.raises(SystemExit):
        check_corpus(["glm2-s50", "glm2-s33"], stats, tids, lambda r, t: pages[t])
