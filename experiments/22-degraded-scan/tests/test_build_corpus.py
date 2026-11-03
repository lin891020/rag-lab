"""Which images each route reads, and how a reply's format is named."""

from build_corpus import PNG06, ROUTES, SCAN, page_dir, reply_format


def test_routes_are_the_planned_ones():
    assert ROUTES == {
        "glm2-L0": ("glm2", "L0"),
        "glm2-L1": ("glm2", "L1"),
        "glm2-L2": ("glm2", "L2"),
        "vision-L1": ("vision", "L1"),
        "vision-L2": ("vision", "L2"),
        "opus-L1": ("opus", "L1"),
        "opus-L2": ("opus", "L2"),
    }


def test_l0_reads_06s_clean_pngs_and_the_others_the_scans():
    assert page_dir("L0") == (PNG06, "png")
    assert page_dir("L1") == (SCAN / "L1", "jpg")
    assert page_dir("L2") == (SCAN / "L2", "jpg")


def test_reply_format():
    assert reply_format("<table><tr><td>1</td></tr></table>") == "html"
    assert reply_format("| a | b |\n|---|---|") == "markdown"
    assert reply_format("a\tb\n1\t2") == "tsv"
    assert reply_format("表五二 需量因數 100") == "plain"


def _img(tmp_path, data=b"img-bytes"):
    """A real (tiny) JPEG whose pixels depend on `data`, so other data = another image."""
    from PIL import Image

    p = tmp_path / "表五二-p0.jpg"
    Image.new("L", (8, 8), sum(data) % 256).save(p, format="JPEG")
    return p


def test_cached_reply_is_reused_only_for_the_same_image(tmp_path, monkeypatch):
    import build_corpus

    calls = []
    monkeypatch.setattr(
        build_corpus,
        "glm_ocr",
        lambda img, prompt: (
            calls.append(img)
            or {
                "content": "<table><tr><td>1</td></tr></table>",
                "done_reason": "stop",
                "eval_count": 9,
            }
        ),
    )
    img, raw = _img(tmp_path), tmp_path / "raw"
    ledger = tmp_path / "ledger.jsonl"
    md, info = build_corpus.read_page("glm2", img, raw, ledger)
    assert md == "| 1 |\n|---|" and info["done_reason"] == "stop" and info["format"] == "html"
    _, again = build_corpus.read_page("glm2", img, raw, ledger)
    assert len(calls) == 1 and again["cached"]
    _img(tmp_path, b"other bytes")  # degrade.py produced another image
    import pytest

    with pytest.raises(SystemExit):
        build_corpus.read_page("glm2", img, raw, ledger)


def test_every_paid_call_is_in_the_ledger_even_a_stopped_one(tmp_path, monkeypatch):
    import build_corpus
    import claude_ocr
    import pytest

    replies = iter(["ok", "stop"])

    def fake(image, media_type):
        if next(replies) == "stop":
            raise claude_ocr.StopError("max_tokens", tokens_in=1000, tokens_out=16000)
        return {
            "content": "<table></table>",
            "tokens_in": 2000,
            "tokens_out": 1000,
            "duration": 1.0,
            "served_model": "claude-opus-5-5",
            "request_id": "r",
        }

    monkeypatch.setattr(claude_ocr, "ocr", fake)
    ledger = tmp_path / "ledger.jsonl"
    build_corpus.read_page("opus", _img(tmp_path), tmp_path / "raw", ledger)
    with pytest.raises(claude_ocr.StopError):
        build_corpus.read_page("opus", _img(tmp_path, b"p2"), tmp_path / "raw2", ledger)
    assert build_corpus.ledger_usd(ledger) == pytest.approx(
        (2000 * 4 + 1000 * 20 + 1000 * 4 + 16000 * 20) / 1e6
    )


def test_opus_stops_before_a_call_once_the_budget_is_spent(tmp_path, monkeypatch):
    import json

    import build_corpus
    import claude_ocr
    import pytest

    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"tokens_in": 0, "tokens_out": 300_000}) + "\n")  # US$6
    monkeypatch.setattr(claude_ocr, "ocr", lambda *a, **k: pytest.fail("must not call"))
    with pytest.raises(SystemExit):
        build_corpus.read_page("opus", _img(tmp_path), tmp_path / "raw", ledger)


def test_check_corpus_rejects_a_partial_or_missing_route(tmp_path):
    import pytest

    from build_corpus import check_corpus

    tids = ["表一", "表二"]
    stats = {"glm2-L1": {"表一": {"pages": 1}, "表二": {"pages": 2}}}
    pages = {("L1", "表一"): 1, ("L1", "表二"): 2}
    check_corpus(["glm2-L1"], stats, tids, lambda lv, t: pages[(lv, t)])
    with pytest.raises(SystemExit):  # a trial run left one page of 表二
        check_corpus(
            ["glm2-L1"],
            {"glm2-L1": {"表一": {"pages": 1}, "表二": {"pages": 1}}},
            tids,
            lambda lv, t: pages[(lv, t)],
        )
    with pytest.raises(SystemExit):  # opus never built
        check_corpus(["glm2-L1", "opus-L1"], stats, tids, lambda lv, t: pages[(lv, t)])
